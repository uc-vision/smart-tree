from dataclasses import dataclass
from typing import List, Optional

import torch
from spconv.pytorch.utils import PointToVoxel
from tensordict import TensorDict

from ..data_types.cloud import Cloud
from ..util.misc import Singleton


@dataclass
class VoxelizedCloud:
    voxel_features: torch.Tensor
    voxel_targets: Optional[TensorDict]
    voxel_coords: torch.Tensor
    voxel_ids: torch.Tensor
    voxel_mask: Optional[torch.Tensor] = None
    filename: Optional[str | List[str]] = None
    input_cloud: Optional[Cloud] = None


@Singleton
class SparseVoxelizer:
    def __init__(
        self,
        voxel_size: float,
        use_xyz: bool,
        use_rgb: bool,
        voxelize_radius: bool,
        voxelize_direction: bool,
        voxelize_class: bool,
        voxelize_mask: bool,
        min_xyz: List,
        max_xyz: List,
        max_num_voxels: int,
        fp_16: bool = False,
        device=torch.device("cpu"),
    ):

        self.voxel_size = voxel_size
        self.use_xyz = use_xyz
        self.use_rgb = use_rgb
        self.voxelize_radius = voxelize_radius
        self.voxelize_direction = voxelize_direction
        self.voxelize_class = voxelize_class
        self.voxelize_mask = voxelize_mask

        self.device = device
        self.fp_16 = fp_16

        self.num_input_features = 0
        if use_xyz:
            self.num_input_features += 3
        if use_rgb:
            self.num_input_features += 3

        self.num_features = self.num_input_features
        if voxelize_radius:
            self.num_features += 1
        if voxelize_class:
            self.num_features += 1
        if voxelize_direction:
            self.num_features += 3
        if voxelize_mask:
            self.num_features += 1

        self.point_to_voxel = PointToVoxel(
            vsize_xyz=[self.voxel_size] * 3,
            coors_range_xyz=[*min_xyz, *max_xyz],
            num_point_features=self.num_features,
            max_num_voxels=max_num_voxels,
            max_num_points_per_voxel=1,
            device=self.device,
        )

    def __call__(
        self,
        cld: Cloud,
    ):

        features = []

        if self.use_xyz:
            features.append(cld.xyz)
        if self.use_rgb:
            features.append(cld.rgb)

        if self.voxelize_radius:
            features.append(cld.radius)
        if self.voxelize_direction:
            features.append(cld.direction)
        if self.voxelize_class:
            features.append(cld.class_l)
        if self.voxelize_mask:
            features.append(cld.mask)

        features = torch.cat(features, dim=1) if len(features) > 0 else features[0]

        voxel_features, voxel_coordinates, num_pts, voxel_ids = (
            self.point_to_voxel.generate_voxel_with_id(features)
        )

        voxel_features = voxel_features.squeeze(1)
        voxel_coordinates = voxel_coordinates.squeeze(1)
        indice = torch.zeros(
            (voxel_coordinates.shape[0], 1),
            dtype=torch.int32,
            device=self.device,
        )

        voxel_coordinates = torch.cat((indice, voxel_coordinates), dim=1)

        dtype = torch.float16 if self.fp_16 else torch.float32

        input_voxel_features = voxel_features[:, : self.num_input_features].to(dtype)

        targets = TensorDict({}, batch_size=[voxel_features.shape[0]])

        index = self.num_input_features

        if self.voxelize_radius:
            targets["radius"] = voxel_features[:, index : index + 1].to(dtype)
            index += 1
        if self.voxelize_direction:
            targets["direction"] = voxel_features[:, index : index + 3].to(dtype)
            index += 3
        if self.voxelize_class:
            targets["class_l"] = voxel_features[:, index : index + 1].to(dtype)
            index += 1

        voxel_mask = voxel_features[:, -1].to(dtype) if self.voxelize_mask else None

        return VoxelizedCloud(
            voxel_features=input_voxel_features,
            voxel_targets=targets,
            voxel_coords=voxel_coordinates,
            voxel_ids=voxel_ids,
            filename=cld.filename,
            voxel_mask=voxel_mask,
        )

    def voxelize_clouds(self, clouds: List[Cloud]):
        return [self.__call__(cloud.to_device(self.device)) for cloud in clouds]

        quit()

        # return TransformedCloud(
        #     voxel_features=input_voxel_features,
        #     voxel_targets=target_voxel_features,
        #     voxel_coords=voxel_coordinates,
        #     voxel_ids=voxel_ids,
        #     filename=cld.filename,
        #     input_cloud=cld,
        #     voxel_mask=voxel_mask,
        # )
