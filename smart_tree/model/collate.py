from typing import List

import torch

from .sparse import sparse_from_batch
from .voxelize import VoxelizedCloud


def inference_collate(clouds: List[VoxelizedCloud]):

    features = []
    coords = []
    voxel_masks = []
    point_ids = []
    input_clouds = []

    for i, voxel_cloud in enumerate(clouds):
        features.append(voxel_cloud.features)
        voxel_cloud.coords[:, 0] = torch.tensor([i], dtype=torch.float32)

        coords.append(voxel_cloud.coords)
        voxel_masks.append(voxel_cloud.voxel_mask)
        point_ids.append(voxel_cloud.point_ids)
        input_clouds.append(voxel_cloud.original_cloud)

    sparse_t = sparse_from_batch(torch.cat(features, dim=0), torch.cat(coords, dim=0))

    return sparse_t, point_ids, input_clouds, voxel_masks
