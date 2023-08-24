from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, rand
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import o3d_viewer
from ..util.queries import skeleton_to_points
from ..util.misc import to_torch, voxel_downsample


@typechecked
@dataclass
class Cloud:
    xyz: TensorType["N", 3]
    rgb: Optional[TensorType["N", 3]] = None
    medial_vector: Optional[TensorType["N", 3]] = None
    class_l: Optional[TensorType["N", 1]] = None

    def __len__(self):
        return self.xyz.shape[0]

    def __str__(self):
        return f"{'*' * 80}\nCloud with {self.xyz.shape[0]} Points.\nMin: {torch.min(self.xyz, 0)[0]}\nMax: {torch.max(self.xyz, 0)[0]}\nDevice:{self.xyz.device}\n{'*' * 80}\n"

    def to_o3d_cld(self):
        cpu_cld = self.to_device("cpu")
        return o3d_cloud(cpu_cld.xyz, colours=cpu_cld.rgb)

    def to_o3d_seg_cld(self, cmap):
        cpu_cld = self.to_device("cpu")
        return o3d_cloud(cpu_cld.xyz, colours=cmap[cpu_cld.class_l.squeeze(1)])

    def filter(cloud: Cloud, mask) -> Cloud:
        mask = mask.to(cloud.xyz.device)
        xyz = cloud.xyz[mask]
        rgb = cloud.rgb[mask] if cloud.rgb is not None else None

        medial_vector = (
            cloud.medial_vector[mask] if cloud.medial_vector is not None else None
        )
        # branch_direction = (
        #     cloud.branch_direction[mask] if cloud.branch_direction is not None else None
        # )
        class_l = cloud.class_l[mask] if cloud.class_l is not None else None
        # branch_ids = cloud.branch_ids[mask] if cloud.branch_ids is not None else None

        return Cloud(
            xyz=xyz,
            rgb=rgb,
            medial_vector=medial_vector,
            # branch_direction=branch_direction,
            class_l=class_l,
            # branch_ids=branch_ids,
        )

    def filter_by_class(self, classes):
        classes = torch.tensor(classes, device=self.class_l.device)
        mask = torch.isin(
            self.class_l,
            classes,
        )
        return self.filter(mask.view(-1))

    def filter_by_skeleton(skeleton: TreeSkeleton, cloud: Cloud, threshold=1.1):
        distances, radii, vectors_ = skeleton_to_points(cloud, skeleton, chunk_size=512)
        mask = distances < radii * threshold
        return filter(cloud, mask)

    def to_device(self, device):
        xyz = self.xyz.to(device)
        rgb = self.rgb.to(device) if self.rgb is not None else None
        medial_vector = (
            self.medial_vector.to(device) if self.medial_vector is not None else None
        )
        # branch_direction = (
        #     self.branch_direction.to(device)
        #     if self.branch_direction is not None
        #     else None
        # )
        class_l = self.class_l.to(device) if self.class_l is not None else None
        # branch_ids = self.branch_ids.to(device) if self.branch_ids is not None else None

        return Cloud(xyz, rgb, medial_vector, class_l)

    def cat(self):
        return torch.cat(
            (
                self.xyz,
                self.rgb,
            ),
            1,
        )

    def view(self, cmap=[]):
        if cmap == []:
            cmap = np.random.rand(self.number_classes, 3)

        cpu_cld = self.to_device("cpu")
        geoms = []

        geoms.append(cpu_cld.to_o3d_cld())
        if cpu_cld.class_l != None:
            geoms.append(self.to_o3d_seg_cld(cmap))

        if cpu_cld.medial_vector != None:
            projected = o3d_cloud(cpu_cld.xyz + cpu_cld.medial_vector, colour=(1, 0, 0))
            geoms.append(projected)
            geoms.append(o3d_lines_between_clouds(cpu_cld.to_o3d_cld(), projected))

        o3d_viewer(geoms)

    def voxel_down_sample(self, voxel_size):
        idx = voxel_downsample(self.xyz, voxel_size)
        return self.filter(idx)

    def scale(self, factor):
        return Cloud(self.xyz * factor, self.rgb)

    def translate(self, xyz):
        return Cloud(self.xyz + xyz.to(self.xyz.device), self.rgb)

    def rotate(self, rot_mat):
        rot_mat = rot_mat.to(self.xyz.dtype)
        return Cloud(torch.matmul(self.xyz, rot_mat.to(self.xyz.device)), self.rgb)

    @property
    def root_idx(self) -> int:
        return torch.argmin(self.xyz[:, 1]).item()

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return torch.max(self.class_l).item() + 1

    @property
    def max_xyz(self) -> torch.Tensor:
        return torch.max(self.xyz, 0)[0]

    @property
    def min_xyz(self) -> torch.Tensor:
        return torch.min(self.xyz, 0)[0]

    @property
    def bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        # defined by centre coordinate, x/2, y/2, z/2
        dimensions = (self.max_xyz - self.min_xyz) / 2
        centre = self.min_xyz + dimensions
        return centre, dimensions

    @property
    def medial_pts(self) -> torch.Tensor:
        return self.xyz + self.medial_vector

    @staticmethod
    def from_numpy(**kwargs) -> "Cloud":
        torch_kwargs = {}

        for key, value in kwargs.items():
            torch_kwargs[key] = torch.tensor(value)

        return Cloud(**torch_kwargs)

    @property
    def radii(self) -> torch.Tensor:
        return self.medial_vector.pow(2).sum(1).sqrt()

    @property
    def direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector)

    @staticmethod
    def from_o3d_cld(cld) -> Cloud:
        return Cloud.from_numpy(xyz=np.asarray(cld.points), rgb=np.asarray(cld.colors))


# @dataclass
# class LabelledCloud(Cloud):
#     vector: torch.Tensor
#     class_l: torch.Tensor

#     @property
#     def number_classes(self):
#         return int(torch.max(self.class_l, 0)[0].item()) + 1

#     @property
#     def cmap(self):
#         return torch.rand(self.number_classes, 3)

#     def filter(self, mask):
#         return LabelledCloud(
#             self.xyz[mask],
#             self.rgb[mask],
#             self.vector[mask],
#             self.class_l[mask],
#         )

#     def filter_by_class(self, classes: List):
#         classes = torch.tensor(classes, device=self.class_l.device)
#         mask = torch.isin(
#             self.class_l,
#             classes,
#         )
#         return self.filter(mask)

#     def view(self, cmap=[]):
#         cmap = cmap if cmap != [] else self.cmap
#         cpu_cld = self.to_device("cpu")
#         input_cld = cpu_cld.to_o3d_cld()
#         segmented_cld = o3d_cloud(cpu_cld.xyz, colours=cmap[cpu_cld.class_l])
#         projected = o3d_cloud(cpu_cld.medial_pts, colour=(1, 0, 0))
#         lines = o3d_lines_between_clouds(input_cld, projected)
#         o3d_viewer([input_cld, projected, lines, segmented_cld])

#     def to_device(self, device):
#         return LabelledCloud(
#             self.xyz.to(device),
#             self.rgb.to(device),
#             self.vector.to(device),
#             self.class_l.to(device),
#         )

#     def cat(self):
#         return torch.cat(
#             (
#                 self.xyz,
#                 self.rgb,
#                 self.vector,
#                 self.class_l.unsqueeze(1),
#             ),
#             1,
#         )

#     def medial_voxel_down_sample(self, voxel_size):
#         idx = voxel_downsample_idxs(self.medial_pts, voxel_size)

#         return self.filter(idx)

#     def to_torch(self):
#         return LabelledCloud(
#             torch.from_numpy(self.xyz),
#             torch.from_numpy(self.rgb),
#             torch.from_numpy(self.vector),
#             torch.from_numpy(self.class_l),
#         )

#     def scale(self, factor):
#         return LabelledCloud(
#             self.xyz * factor, self.rgb, self.vector * factor, self.class_l
#         )

#     def translate(self, xyz):
#         return LabelledCloud(self.xyz + xyz, self.rgb, self.vector, self.class_l)

#     def rotate(self, rot_mat):
#         new_xyz = torch.matmul(self.xyz, rot_mat.to(self.xyz.dtype))
#         new_vectors = torch.matmul(self.vector, rot_mat.to(self.vector.dtype))

#         return LabelledCloud(new_xyz, self.rgb, new_vectors, self.class_l)

#     @property
#     def radii(self):
#         return self.vector.pow(2).sum(1).sqrt()

#     @property
#     def direction(self):
#         return F.normalize(self.vector)

#     @property
#     def medial_pts(self):
#         return self.xyz + self.vector

#     @staticmethod
#     def from_numpy(xyz, rgb, vector, class_l):
#         return LabelledCloud(
#             torch.from_numpy(xyz),  # float64 -> these data types are stupid...
#             torch.from_numpy(rgb),  # float64
#             torch.from_numpy(vector),  # float32
#             torch.from_numpy(class_l),  # int64
#         )
