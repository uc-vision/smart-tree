from functools import partial
from typing import List, Union

import spconv.pytorch as spconv
import torch
from py_structs.torch import map_tensors
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


def sparse_voxelize(
    cld: Cloud | LabelledCloud,
    voxel_size: float,
    input_features: List[str],
    target_features: List[str] = [],
):
    xyzmin, xyzmax = cld.bounding_box

    all_feats = []
    len_input_feats = len_target_feats = len_mask_feats = 0

    if getattr(cld, "loss_mask", None) is None:
        cld.loss_mask = torch.ones((len(cld), 1), dtype=torch.bool, device=cld.device)

    for feat in input_features + target_features + ["loss_mask"]:
        assert getattr(cld, feat, None) is not None, f"{feat} is None"
        assert isinstance(getattr(cld, feat), torch.Tensor), f"{feat} not a tensor"

        feat_shape = getattr(cld, feat).shape[1]
        all_feats.append(getattr(cld, feat))

        if feat in input_features:
            len_input_feats += feat_shape
        if feat in target_features:
            len_target_feats += feat_shape
        if feat == "loss_mask":
            len_mask_feats += feat_shape

    if len(all_feats) > 1:
        all_feats = torch.cat(all_feats, dim=1).contiguous()

    voxel_gen = PointToVoxel(
        vsize_xyz=[voxel_size] * 3,
        coors_range_xyz=[
            xyzmin[0],
            xyzmin[1],
            xyzmin[2],
            xyzmax[0],
            xyzmax[1],
            xyzmax[2],
        ],
        num_point_features=all_feats.shape[1],
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    feats_voxelized, coords, num_pts, voxel_id = voxel_gen.generate_voxel_with_id(
        all_feats
    )

    feats_voxelized = feats_voxelized.squeeze(1)
    coords = coords.squeeze(1)

    input_feats = feats_voxelized[:, :len_input_feats]
    target_feats = feats_voxelized[
        :,
        len_input_feats : len_input_feats + len_target_feats,
    ]
    mask = feats_voxelized[:, -len_mask_feats:] != 0

    indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    coords = torch.cat((indice, coords), dim=1)

    if target_feats != []:
        return ((input_feats, target_feats), coords, mask, cld.filename)

    return input_feats, coords, mask, cld.filename
