import json
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.utils.data
from typeguard import typechecked

from dataclasses import asdict

from ..data_types.tube import collate_tubes

from tqdm import tqdm

from ..data_types.cloud import Cloud, LabelledCloud
from ..util.file import CloudLoader, Skeleton_and_Cloud_Loader
from .augmentations import AugmentationPipeline

from .contraction import *

# import geometry_grid.torch_geometry as torch_geom


@typechecked
class Dataset:
    def __init__(
        self,
        json_path: Path | str,
        directory: Path | str,
        mode: Literal["train", "validation", "test", "capture", "unlabelled"],
        transform: Optional[callable] = None,
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = False,
        self_consistency: bool = False,
        loader=CloudLoader(),
        device=torch.device("cuda:0"),
        label=False,
    ):
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.self_consistency = self_consistency
        self.device = device
        self.label = label

        assert Path(json_path).is_file(), f"JSON path is invalid: '{json_path}'"
        assert Path(directory).is_dir(), f"Directory path is invalid: {directory}"

        json_data = json.load(open(json_path))

        if self.mode == "train":
            tree_paths = json_data["train"]
        elif self.mode == "validation":
            tree_paths = json_data["validation"]
        elif mode == "test":
            tree_paths = json_data["test"]
        elif mode == "capture":
            tree_paths = json_data["capture"]

        self.full_paths = [Path(f"{directory}/{p}") for p in tree_paths]
        invalid_paths = [path for path in self.full_paths if not path.exists()]
        assert not invalid_paths, f"Missing {len(invalid_paths)} files: {invalid_paths}"

        self.loader = loader

        self.cache = {}
        self.cache_data() if cache else None

    def cache_data(self):
        for path in tqdm(self.full_paths, desc="Caching Dataset", leave=False):
            data = self.loader.load(path)
            self.cache[path] = self.loader.load(path).pin_memory()

    def load(self, filename) -> Cloud | LabelledCloud:
        if self.cache is None:
            return self.loader.load(filename)

        if filename not in self.cache:
            # self.cache[filename] = [x.pin_memory() for x in self.loader.load(filename)]
            self.cache[filename] = self.loader.load(filename)

        return self.cache[filename]

    def __getitem__(self, idx):
        data = self.load(self.full_paths[idx])
        try:
            return self.process(data)
        except Exception:
            print(f"Exception processing {self.full_paths[idx]}")
            raise

    def __len__(self):
        return len(self.full_paths)

    def process(self, data) -> Any:
        if self.label:
            skeleton, cld = data
            # skeleton.repair()

            torch_tubes = collate_tubes(skeleton.to_tubes())

            tubes = {
                k: x.to(dtype=torch.float32, device=self.device)
                for k, x in asdict(torch_tubes).items()
            }

            segments = torch_geom.Segment(tubes["a"], tubes["b"])
            radii = torch.stack((tubes["r1"], tubes["r2"]), -1).squeeze(0)

            tubes = torch_geom.Tube(segments, radii.reshape(-1, 2))
            bounds = tubes.bounds.union_all()

            points = (cld.xyz).to(dtype=torch.float32, device=self.device)
            points = morton_sort(points, n=256)

            tube_grid = CountedGrid.from_torch(
                Grid.fixed_size(bounds, (16, 16, 16)), tubes
            )

            # point_grid = DynamicGrid.from_torch(
            #     Grid.fixed_size(bounds, (64, 64, 64)), torch_geom.Point(points))
            _, idx = min_query(tube_grid, points, 0.2, relative_distance)

            to_axis = to_medial_axis(segments[idx], points)

            cld = cld.to_labelled_cloud(medial_vector=to_axis)

        cld = data.to_device(self.device)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        if self.transform != None:
            cld = self.transform(cld)

        return cld


class SingleTreeInference:
    def __init__(
        self,
        cloud: Cloud,
        block_size: float = 4.0,
        buffer_size: float = 0.4,
        min_points: int = 20,
        augmentation: Optional[callable] = None,
        transform: Optional[callable] = None,
    ):
        self.cloud = cloud
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.augmentation = augmentation
        self.transform = transform

        xyz_quantized = torch.div(self.cloud.xyz, block_size, rounding_mode="floor")
        block_ids, pnt_counts = torch.unique(xyz_quantized, return_counts=True, dim=0)

        valid_block_ids = block_ids[pnt_counts > min_points]
        self.block_centres = (valid_block_ids * block_size) + (block_size / 2)

    def __getitem__(self, idx) -> Any:
        centre = self.block_centres[idx]

        xyzmin = centre - (self.block_size / 2)
        xyzmax = centre + (self.block_size / 2)

        block_max_xyz = xyzmax + self.buffer_size
        block_min_xyz = xyzmin - self.buffer_size

        block_mask = (self.cloud.xyz > block_min_xyz) & (self.cloud.xyz < block_max_xyz)
        block_mask = torch.all(block_mask, dim=1)

        block_cld = self.cloud.filter(block_mask)

        buffer_mask = (block_cld.xyz > xyzmin) & (block_cld.xyz < xyzmax)
        buffer_mask = torch.all(buffer_mask, dim=1).unsqueeze(1)

        cld = block_cld.to_labelled_cloud(loss_mask=buffer_mask)

        if self.augmentation:
            cld = self.augmentation(cld)

        if self.transform:
            cld = self.transform(cld)

        return cld

    def __len__(self):
        return len(self.block_centres)


@typechecked
class ContractionDataset(Dataset):
    def __init__(
        self,
        json_path: Path | str,
        directory: Path | str,
        mode: Literal["train", "validation", "test", "capture", "unlabelled"],
        transform: Optional[callable] = None,
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = False,
        self_consistency: bool = False,
        loader=Skeleton_and_Cloud_Loader(),
        device=torch.device("cuda:0"),
        label=True,
    ):
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.self_consistency = self_consistency
        self.device = device
        self.label = label

        assert Path(json_path).is_file(), f"JSON path is invalid: '{json_path}'"
        assert Path(directory).is_dir(), f"Directory path is invalid: {directory}"

        json_data = json.load(open(json_path))

        if self.mode == "train":
            tree_paths = json_data["train"]
        elif self.mode == "validation":
            tree_paths = json_data["validation"]
        elif mode == "test":
            tree_paths = json_data["test"]
        elif mode == "capture":
            tree_paths = json_data["capture"]

        self.full_paths = [Path(f"{directory}/{p}") for p in tree_paths]
        invalid_paths = [path for path in self.full_paths if not path.exists()]
        assert not invalid_paths, f"Missing {len(invalid_paths)} files: {invalid_paths}"

        self.tube_grid_cache = {}
        # self.

        self.loader = loader

        self.cache = {}
        self.cache_data() if cache else None

    def cache_data(self):
        for path in tqdm(self.full_paths, desc="Caching Dataset", leave=False):
            self.cache[path] = self.loader.load(path).pin_memory()

    def load(self, filename) -> Cloud | LabelledCloud:
        if self.cache is None:
            return self.loader.load(filename)

        if filename not in self.cache:
            self.cache[filename] = [x.pin_memory() for x in self.loader.load(filename)]

        return self.cache[filename]

    def __getitem__(self, idx):
        data = self.load(self.full_paths[idx])
        try:
            return self.process(data)
        except Exception:
            print(f"Exception processing {self.full_paths[idx]}")
            raise

    def __len__(self):
        return len(self.full_paths)

    def process(self, data) -> Any:
        if self.label:
            skeleton, cld = data
            # skeleton.repair()

            torch_tubes = collate_tubes(skeleton.to_tubes())

            tubes = {
                k: x.to(dtype=torch.float32, device=self.device)
                for k, x in asdict(torch_tubes).items()
            }

            segments = torch_geom.Segment(tubes["a"], tubes["b"])
            radii = torch.stack((tubes["r1"], tubes["r2"]), -1).squeeze(0)

            tubes = torch_geom.Tube(segments, radii.reshape(-1, 2))
            bounds = tubes.bounds.union_all()

            points = (cld.xyz).to(dtype=torch.float32, device=self.device)
            points = morton_sort(points, n=256)

            tube_grid = CountedGrid.from_torch(
                Grid.fixed_size(bounds, (16, 16, 16)), tubes
            )

            # point_grid = DynamicGrid.from_torch(
            #     Grid.fixed_size(bounds, (64, 64, 64)), torch_geom.Point(points))
            _, idx = min_query(tube_grid, points, 0.2, relative_distance)

            to_axis = to_medial_axis(segments[idx], points)

            cld = cld.to_labelled_cloud(medial_vector=to_axis)

        cld = cld.to_device(self.device)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        if self.transform != None:
            cld = self.transform(cld)

        return cld
