from functools import partial
from typing import List

import spconv.pytorch as spconv
import torch
from py_structs.torch import map_tensors
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


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
    target_features: List[str],
):
    xyzmin, xyzmax = cld.bounding_box

    len_input_feats = 0
    len_target_feats = 0
    len_mask_feats = 0

    all_feats = []

    masks = ["loss_mask", "vector_mask"]

    for feat in input_features + target_features + masks:
        assert hasattr(cld, feat), f"Cloud does not have {feat} attribute"
        assert getattr(cld, feat).dtype == torch.tensor, f"{feat} is not a tensor"

        if feat in input_feats:
            len_input_feats += getattr(cld, feat).shape[1]
        if feat in target_features:
            len_target_feats += getattr(cld, feat).shape[1]
        if feat in masks:
            len_mask_feats += getattr(cld, feat).shape[1]
        all_feats.append(getattr(cld, feat))

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
        num_point_features=all_feats.shape[0],
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

    if target_features is not None:
        target_features = feats_voxelized[
            :, len_input_feats : len_input_feats + len_target_feats
        ]

    # indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    # coords = torch.cat((indice, coords), dim=1)
    # feats = feats.squeeze(1)
    # coords = coords.squeeze(1)

    # mask = torch.ones((feats.shape[0], 1), dtype=torch.int32)

    # if target_feats is None:
    #     return [feats, coords, mask, cld.filename]

    # else:
    #     return [
    #         (feats[:, : input_feats.shape[1]], feats[:, input_feats.shape[1] :]),
    #         coords,
    #         mask,
    #         cld.filename,
    #     ]
