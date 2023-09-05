from functools import partial
from typing import List

import spconv.pytorch as spconv
import torch
from py_structs.torch import map_tensors
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


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


def sparse_from_batch(features, coordinates, device):
    batch_size = features.shape[0]

    features = features.to(device)
    coordinates = coordinates.to(device)

    values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

    return spconv.SparseConvTensor(
        features,
        coordinates.int(),
        values[1:],
        batch_size=batch_size,
    )


def get_batch(dataloader, device, fp_16=False):
    for (feats, target_feats), coords, mask, filenames in dataloader:
        if fp_16:
            feats = feats.half()
            target_feats = target_feats.half()
            coords = coords.half()

        sparse_input = sparse_from_batch(
            feats,
            coords,
            device=device,
        )
        targets = map_tensors(
            target_feats,
            partial(
                torch.Tensor.to,
                device=device,
            ),
        )

        yield sparse_input, targets, mask, filenames


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

    return (
        [input_feats, coords, mask, cld.filename]
        if target_feats != []
        else [(input_feats, target_feats), coords, mask, cld.filename]
    )
