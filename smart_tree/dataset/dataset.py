import json
from pathlib import Path
from typing import List, Literal, Optional, Any

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from typeguard import typechecked

from ..data_types.cloud import (
    Cloud,
    CloudLoader,
    LabelledCloud,
    convert_cloud_to_labelled_cloud,
)
from .augmentations import AugmentationPipeline


@typechecked
class Dataset:
    def __init__(
        self,
        json_path: Path | str,
        directory: Path | str,
        mode: Literal["train", "validation", "test", "unlabelled"],
        transform: Optional[callable] = None,
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = False,
        device=torch.device("cuda:0"),
    ):
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.device = device

        assert Path(json_path).is_file(), f"JSON path is invalid: '{json_path}'"
        assert Path(directory).is_dir(), f"Directory path is invalid: {directory}"

        json_data = json.load(open(json_path))

        if self.mode == "train":
            tree_paths = json_data["train"]
        elif self.mode == "validation":
            tree_paths = json_data["validation"]
        elif mode == "test":
            tree_paths = json_data["test"]

        self.full_paths = [Path(f"{directory}/{p}") for p in tree_paths]
        invalid_paths = [path for path in self.full_paths if not path.exists()]
        assert not invalid_paths, f"Missing {len(invalid_paths)} files: {invalid_paths}"

        self.cache = {} if cache else None
        self.cloud_loader = CloudLoader()

    def load(self, filename) -> Cloud | LabelledCloud:
        if self.cache is None:
            return self.cloud_loader.load(filename)

        if filename not in self.cache:
            self.cache[filename] = self.cloud_loader.load(filename).pin_memory()

        return self.cache[filename]

    def __getitem__(self, idx):
        cld = self.load(self.full_paths[idx])
        try:
            return self.process_cloud(cld)
        except Exception:
            print(f"Exception processing {cld}")
            raise

    def __len__(self):
        return len(self.full_paths)

    def process_cloud(self, cld: Cloud | LabelledCloud) -> Any:
        data = cld.to_device(self.device)

        if self.augmentation != None:
            data = self.augmentation(data)

        if self.transform != None:
            data = self.transform(data)

        return data


class SingleTreeInference:
    def __init__(
        self,
        cloud: Cloud,
        block_size: float = 4.0,
        buffer_region: float = 0.4,
        min_points: int = 20,
        augmentation: Optional[callable] = None,
        transform: Optional[callable] = None,
    ):
        self.cloud = cloud
        self.block_size = block_size
        self.buffer_region = buffer_region
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

        block_mask = torch.all(
            (self.cloud.xyz > xyzmin - self.buffer_region)
            & (self.cloud.xyz < xyzmax + self.buffer_region),
            dim=1,
        )

        block_cld = self.cloud.filter(block_mask)

        buffer_mask = torch.all(
            (block_cld.xyz > xyzmin) & (block_cld.xyz < xyzmax),
            dim=1,
        ).unsqueeze(1)

        cld = convert_cloud_to_labelled_cloud(block_cld, loss_mask=buffer_mask)

        if self.augmentation:
            cld = self.augmentation(cld)

        if self.transform:
            cld = self.transform(cld)

        return cld

    def __len__(self):
        return len(self.block_centres)


def load_dataloader(
    cloud: Cloud,
    voxel_size: float,
    block_size: float,
    buffer_size: float,
    num_workers: float,
    batch_size: float,
):
    dataset = SingleTreeInference(cloud, voxel_size, block_size, buffer_size)

    return DataLoader(dataset, batch_size, num_workers, collate_fn=batch_collate)
