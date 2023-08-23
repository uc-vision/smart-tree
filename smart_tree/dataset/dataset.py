import json
import time
from pathlib import Path
from typing import List

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.utils.data
from spconv.pytorch.utils import PointToVoxel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_types.cloud import Cloud, LabelledCloud
from ..model.sparse import batch_collate, sparse_quantize
from ..util.file import load_data_npz
from ..util.math.maths import cube_filter, np_normalized, torch_normalized
from ..util.visualizer.view import o3d_viewer
from .augmentations import AugmentationPipeline


class TreeDataset:
    def __init__(
        self,
        voxel_size: int,
        json_path: Path,
        directory: Path,
        mode: str,
        blocking: bool,
        block_size: float,
        buffer_size: float,
        augmentation=None,
    ):
        self.voxel_size = voxel_size
        self.mode = mode
        self.blocking = blocking
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.augmentation = AugmentationPipeline.from_cfg(augmentation)
        self.directory = directory

        json_data = json.load(open(json_path))

        if self.mode == "train":
            self.tree_paths = json_data["train"]
        elif self.mode == "validation":
            self.tree_paths = json_data["validation"]

    def __getitem__(self, idx):
        labelled_cld, _ = load_data_npz(f"{self.directory}/{self.tree_paths[idx]}")

        labelled_cld = labelled_cld.to_device(torch.device("cuda"))

        if self.augmentation != None:
            labelled_cld = self.augmentation(labelled_cld)

        if self.blocking:
            block_center_idx = torch.randint(
                labelled_cld.xyz.shape[0], size=(1,), device=labelled_cld.xyz.device
            )
            block_center = labelled_cld.xyz[block_center_idx].reshape(-1)
            block_filter = cube_filter(
                labelled_cld.xyz,
                block_center,
                self.block_size + (self.buffer_size * 2),
            )
            labelled_cld = labelled_cld.filter(block_filter)

        xyzmin, _ = torch.min(labelled_cld.xyz, axis=0)
        xyzmax, _ = torch.max(labelled_cld.xyz, axis=0)
        make_voxel_gen = time.time()

        data = labelled_cld.cat()

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
            dtype=torch.int32,
            device=feats.device,
        )
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)
        loss_mask = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)

        if self.blocking:
            loss_mask = cube_filter(feats[:, :3], block_center, self.block_size)

        return feats.float(), coords.int(), loss_mask

    def __len__(self):
        return len(self.tree_paths)


class SingleTreeInference:
    def __init__(
        self,
        cloud: Cloud,
        voxel_size: float,
        block_size: float = 4,
        buffer_size: float = 0.4,
        min_points=20,
        device=torch.device("cuda:0"),
    ):
        self.cloud = cloud

        self.voxel_size = voxel_size
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.min_points = min_points
        self.device = device

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
            cloud.cat().contiguous()
        )  # FEATURES, COORD, _, ID

        indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)

        mask = cube_filter(feats[:, :3], block_centre, self.block_size)

        return feats, coords, mask

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
