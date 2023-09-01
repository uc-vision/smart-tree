import json
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.utils.data
from spconv.pytorch.utils import PointToVoxel
from torch.utils.data import DataLoader
from tqdm import tqdm
from typeguard import typechecked

from ..data_types.cloud import Cloud, CloudLoader, LabelledCloud
from ..model.sparse import batch_collate
from ..util.maths import cube_filter
from ..util.misc import at_least_2d
from .augmentations import AugmentationPipeline


@typechecked
class Dataset:
    def __init__(
        self,
        json_path: Path | str,
        directory: Path | str,
        mode: Literal["train", "validation", "test", "unlabelled"],
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = False,
        device=torch.device("cuda:0"),
    ):
        self.mode = mode
        self.augmentation = augmentation
        self.directory = directory
        self.device = device

        assert Path(json_path).is_file(), f"No json metadata at '{json_path}'"
        json_data = json.load(open(json_path))

        if self.mode == "train":
            tree_paths = json_data["train"]
        elif self.mode == "validation":
            tree_paths = json_data["validation"]
        elif mode == "test":
            tree_paths = json_data["test"]

        # Check the value of self.directory
        assert Path(self.directory).is_dir(), "Directory path is invalid"

        self.full_paths = [Path(f"{self.directory}/{p}") for p in tree_paths]
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
            return self.process_cloud(cld, self.full_paths[idx])
        except Exception:
            print(f"Exception processing {cld}")
            raise

    def __len__(self):
        return len(self.full_paths)

    def process_cloud(self, cld: Cloud, filename: str) -> Cloud | LabelledCloud:
        cld = cld.to_device(self.device)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        return cld

        if cld.input_features is None:
            raise ValueError(f"Missing input features {filename}")

        if self.mode != "unlabelled" and cld.target_features is None:
            raise ValueError(f"Missing target features {filename}")

        return cld

        # block_center_idx = torch.randint(
        #     cld.xyz.shape[0],
        #     size=(1,),
        #     device=cld.xyz.device,
        # )
        # block_center = cld.xyz[block_center_idx].reshape(-1)
        # block_filter = cube_filter(
        #     cld.xyz,
        #     block_center,
        #     4 + (0.4 * 2),
        # )
        # cld = cld.filter(block_filter)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        xyzmin, _ = torch.min(cld.xyz, axis=0)
        xyzmax, _ = torch.max(cld.xyz, axis=0)

        input_feats = torch.cat(
            [at_least_2d(getattr(cld, attr)) for attr in self.input_features],
            dim=1,
        )

        if cld.is_labelled:
            target_features = torch.cat(
                [at_least_2d(getattr(cld, attr)) for attr in self.target_features],
                dim=1,
            )

            data = torch.cat([input_feats, target_features], dim=1)
        else:
            data = input_feats

        assert (
            data.shape[0] > 0
        ), f"Empty cloud after augmentation: {self.tree_paths[idx]}"

        surface_voxel_generator = PointToVoxel(
            vsize_xyz=[self.voxel_size] * 3,
            coors_range_xyz=[
                xyzmin[0],
                xyzmin[1],
                xyzmin[2],
                xyzmax[0],
                xyzmax[1],
                xyzmax[2],
            ],
            num_point_features=data.shape[1],
            max_num_voxels=data.shape[0],
            max_num_points_per_voxel=1,
            device=data.device,
        )

        feats, coords, _, _ = surface_voxel_generator.generate_voxel_with_id(data)

        indice = torch.zeros(
            (coords.shape[0], 1),
            dtype=coords.dtype,
            device=feats.device,
        )
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)
        loss_mask = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
        # loss_mask = cube_filter(feats[:, :3], block_center, 4)

        input_feats = feats[:, : input_feats.shape[1]]
        target_feats = feats[:, input_feats.shape[1] :]

        return (input_feats, target_feats), coords, loss_mask, filename


class SingleTreeInference:
    def __init__(
        self,
        cloud: Cloud,
        voxel_size: float,
        block_size: float = 4,
        buffer_size: float = 0.4,
        min_points=20,
        file_name=None,
        device=torch.device("cuda:0"),
    ):
        self.cloud = cloud

        self.voxel_size = voxel_size
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.min_points = min_points
        self.device = device
        self.file_name = file_name

        self.compute_blocks()

    def compute_blocks(self):
        self.xyz_quantized = torch.div(
            self.cloud.xyz, self.block_size, rounding_mode="floor"
        )
        self.block_ids, pnt_counts = torch.unique(
            self.xyz_quantized, return_counts=True, dim=0
        )

        # Remove blocks that have less than specified amount of points...
        self.block_ids = self.block_ids[pnt_counts > self.min_points]
        self.block_centres = (self.block_ids * self.block_size) + (self.block_size / 2)

        self.clouds: List[Cloud] = []

        for centre in tqdm(self.block_centres, desc="Computing blocks...", leave=False):
            mask = cube_filter(
                self.cloud.xyz,
                centre,
                self.block_size + (self.buffer_size * 2),
            )
            block_cloud = self.cloud.filter(mask).to_device(torch.device("cpu"))

            self.clouds.append(block_cloud)

        self.block_centres = self.block_centres.to(torch.device("cpu"))

    def __getitem__(self, idx):
        block_centre = self.block_centres[idx]
        cloud: Cloud = self.clouds[idx]

        xyzmin, _ = torch.min(cloud.xyz, axis=0)
        xyzmax, _ = torch.max(cloud.xyz, axis=0)

        surface_voxel_generator = PointToVoxel(
            vsize_xyz=[self.voxel_size] * 3,
            coors_range_xyz=[
                xyzmin[0],
                xyzmin[1],
                xyzmin[2],
                xyzmax[0],
                xyzmax[1],
                xyzmax[2],
            ],
            num_point_features=6,
            max_num_voxels=len(cloud),
            max_num_points_per_voxel=1,
        )

        feats, coords, _, voxel_id_tv = surface_voxel_generator.generate_voxel_with_id(
            torch.cat((cloud.xyz, cloud.rgb), dim=1).contiguous()
        )  #

        indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)

        mask = cube_filter(feats[:, :3], block_centre, self.block_size)

        return feats, coords, mask, self.file_name

    def __len__(self):
        return len(self.clouds)


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
