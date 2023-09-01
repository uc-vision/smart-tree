from typing import List

import torch
from spconv.pytorch.utils import PointToVoxel

from ..data_types.cloud import Cloud


def collate_clds_to_spconv_tensors(
    clds: List[Cloud],
    input_feats_keys=["xyz"],
    target_feats_keys=["class_l"],
    voxel_size: float = 1,
):
    collated_input_feats = []
    collated_target_feats = []

    collated_coords = []

    for cld in clds:
        cld.view()

        xyzmin, xyzmax = cld.bounding_box

        feats_dicts = {
            key: getattr(cld, key) for key in input_feats_keys + target_feats_keys
        }
        feats = torch.cat(feats_dicts.values(), dim=1)

        voxel_generator = PointToVoxel(
            vsize_xyz=[voxel_size] * 3,
            coors_range_xyz=[
                xyzmin[0],
                xyzmin[1],
                xyzmin[2],
                xyzmax[0],
                xyzmax[1],
                xyzmax[2],
            ],
            num_point_features=feats.shape[1],
            max_num_voxels=feats.shape[0],
            max_num_points_per_voxel=1,
            device=cld.device,
        )

        feats, coords, num_points, id = voxel_generator.generate_voxel_with_id(feats)

        indice = torch.zeros(
            (coords.shape[0], 1), dtype=coords.dtype, device=feats.device
        )
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)
        loss_mask = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
        # loss_mask = cube_filter(feats[:, :3], block_center, 4)

        input_feats = feats[:, : input_feats.shape[1]]
        target_feats = feats[:, input_feats.shape[1] :]

        return (input_feats, target_feats), coords, loss_mask, filename


def batch_collate(batch):
    """Custom Batch Collate Function for Sparse Data..."""

    batch_feats, batch_coords, batch_mask, fn = zip(*batch)

    for i, coords in enumerate(batch_coords):
        coords[:, 0] = torch.tensor([i], dtype=torch.float32)

    if isinstance(batch_feats[0], tuple):
        input_feats, target_feats = tuple(zip(*batch_feats))

        input_feats, target_feats, coords, mask = [
            torch.cat(x) for x in [input_feats, target_feats, batch_coords, batch_mask]
        ]

        return [(input_feats, target_feats), coords, mask, fn]

    feats, coords, mask = [
        torch.cat(x) for x in [batch_feats, batch_coords, batch_mask]
    ]

    return [feats, coords, mask, fn]
