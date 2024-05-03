from dataclasses import dataclass
from typing import List, Optional

import torch
from spconv.pytorch.utils import PointToVoxel
from tensordict import TensorDict

from ..data_types.cloud import Cloud


@dataclass
class TransformedCloud:
    voxel_features: torch.Tensor
    voxel_targets: Optional[TensorDict]
    voxel_coords: torch.Tensor
    voxel_ids: torch.Tensor
    voxel_mask: Optional[torch.Tensor] = None

    filename: Optional[str | List[str]] = None
    input_cloud: Optional[Cloud] = None


def sparse_voxelize(
    cld: Cloud,
    voxel_size: float,
    use_xyz: bool,
    use_rgb: bool,
    voxelize_radius: bool,
    voxelize_direction: bool,
    voxelize_class: bool,
    voxelize_mask: bool,
    fp_16: bool = False,
):

    features = []

    input_feat_size = 0

    if use_xyz:
        features.append(cld.xyz)
        input_feat_size += 3
    if use_rgb:
        features.append(cld.rgb)
        input_feat_size += 3

    if voxelize_radius:
        features.append(cld.radius)
    if voxelize_direction:
        features.append(cld.direction)
    if voxelize_class:
        features.append(cld.class_l)
    if voxelize_mask:
        features.append(cld.mask)

    if len(features) > 0:
        features = torch.cat(features, dim=1)
    else:
        features = features[0]

    voxel_gen = PointToVoxel(
        vsize_xyz=[voxel_size] * 3,
        coors_range_xyz=[*cld.min_xyz, *cld.max_xyz],
        num_point_features=features.shape[1],
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    voxel_features, voxel_coordinates, num_pts, voxel_ids = (
        voxel_gen.generate_voxel_with_id(features)
    )

    voxel_features = voxel_features.squeeze(1)
    voxel_coordinates = voxel_coordinates.squeeze(1)
    indice = torch.zeros(
        (voxel_coordinates.shape[0], 1),
        dtype=torch.int32,
        device=cld.device,
    )

    voxel_coordinates = torch.cat((indice, voxel_coordinates), dim=1)

    dtype = torch.float16 if fp_16 else torch.float32

    input_voxel_features = voxel_features[:, :input_feat_size].to(dtype)

    targets = TensorDict({}, batch_size=[voxel_features.shape[0]])

    index = input_feat_size
    if voxelize_radius:
        targets["radius"] = voxel_features[:, index : index + 1].to(dtype)
        index += 1
    if voxelize_direction:
        targets["direction"] = voxel_features[:, index : index + 3].to(dtype)
        index += 3
    if voxelize_class:
        targets["class_l"] = voxel_features[:, index : index + 1].to(dtype)
        index += 1

    voxel_mask = voxel_features[:, -1].to(dtype) if voxelize_mask else None

    target_voxel_features = targets

    return TransformedCloud(
        voxel_features=input_voxel_features,
        voxel_targets=target_voxel_features,
        voxel_coords=voxel_coordinates,
        voxel_ids=voxel_ids,
        filename=cld.filename,
        input_cloud=cld,
        voxel_mask=voxel_mask,
    )
