from smart_tree.data_types.cloud import Cloud, LabelledCloud
import torch

from typing import Union, List
from py_structs.torch import map_tensors
from functools import partial
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.util import (
    get_properties,
    cat_tensor_dict,
    cat_tensor_properties,
)

import spconv.pytorch as spconv


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

    input_feats = cat_tensor_properties(cld, input_features)
    target_feats = cat_tensor_properties(cld, target_features)

    if target_feats is None:
        all_feats = input_feats
    else:
        all_feats = torch.cat([input_feats, target_feats], dim=1)

    all_feats = all_feats.contiguous()
    num_point_feats = all_feats.shape[1]

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
        num_point_features=num_point_feats,
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    (
        feats,
        coords,
        num_pts,
        voxel_id_tv,
    ) = voxel_gen.generate_voxel_with_id(all_feats)

    indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    coords = torch.cat((indice, coords), dim=1)
    feats = feats.squeeze(1)
    coords = coords.squeeze(1)

    mask = torch.ones((feats.shape[0], 1), dtype=torch.int32)

    if target_feats is None:
        return [feats, coords, mask, cld.filename]

    else:
        return [
            (feats[:, : input_feats.shape[1]], feats[:, input_feats.shape[1] :]),
            coords,
            mask,
            cld.filename,
        ]
