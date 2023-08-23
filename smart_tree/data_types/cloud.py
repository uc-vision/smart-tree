from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, rand
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from ..util.math.queries import skeleton_to_points
from ..util.mesh.geometries import o3d_cloud, o3d_lines_between_clouds
from ..util.misc import to_torch, voxel_downsample
from ..util.visualizer.view import o3d_viewer


@typechecked
@dataclass
class Cloud:
    xyz: TensorType["N", 3]
    rgb: TensorType["N", 3]

    def __len__(self):
        return self.xyz.shape[0]

    def __str__(self):
        return f"{'*' * 80}\nCloud with {self.xyz.shape[0]} Points.\nMin: {torch.min(self.xyz, 0)[0]}\nMax: {torch.max(self.xyz, 0)[0]}\nDevice:{self.xyz.device}\n{'*' * 80}\n"

    def to_labelled_cld(self, radii, direction, class_l) -> LabelledCloud:
        return LabelledCloud(self.xyz, self.rgb, radii * direction, class_l)

    def to_o3d_cld(self):
        cpu_cld = self.to_device("cpu")
        return o3d_cloud(cpu_cld.xyz, colours=cpu_cld.rgb)

    def filter(self, mask):
        return Cloud(self.xyz[mask], self.rgb[mask])

    def filter_by_skeleton(skeleton: TreeSkeleton, cloud: Cloud, threshold=1.1):
        distances, radii, vectors_ = skeleton_to_points(cloud, skeleton, chunk_size=512)
        mask = distances < radii * threshold
        return filter(cloud, mask)

    def to_device(self, device):
        return Cloud(self.xyz.to(device), self.rgb.to(device))

    def cat(self):
        return torch.cat(
            (
                self.xyz,
                self.rgb,
            ),
            1,
        )

    def view(self):
        o3d_viewer([self.to_o3d_cld()])

    def voxel_down_sample(self, voxel_size):
        idx = voxel_downsample(self.xyz, voxel_size)
        return self.filter(idx)

    def scale(self, factor):
        return Cloud(self.xyz * factor, self.rgb)

    def translate(self, xyz):
        return Cloud(self.xyz + xyz, self.rgb)

    def rotate(self, rot_mat):
        rot_mat = rot_mat.to(self.xyz.dtype)
        return Cloud(torch.matmul(self.xyz, rot_mat.to(self.xyz.device)), self.rgb)

    @property
    def root_idx(self) -> int:
        return torch.argmin(self.xyz[:, 1]).item()

    @staticmethod
    def from_numpy(xyz, rgb, device=torch.device("cpu")) -> Cloud:
        return Cloud(
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
        ).to_device(device)

    @staticmethod
    def from_o3d_cld(cld) -> Cloud:
        return Cloud.from_numpy(xyz=np.asarray(cld.points), rgb=np.asarray(cld.colors))


@dataclass
class LabelledCloud(Cloud):
    vector: torch.Tensor
    class_l: torch.Tensor

    @property
    def number_classes(self):
        return int(torch.max(self.class_l, 0)[0].item()) + 1

    @property
    def cmap(self):
        return torch.rand(self.number_classes, 3)

    def filter(self, mask):
        return LabelledCloud(
            self.xyz[mask],
            self.rgb[mask],
            self.vector[mask],
            self.class_l[mask],
        )

    def filter_by_class(self, classes: List):
        classes = torch.tensor(classes, device=self.class_l.device)
        mask = torch.isin(
            self.class_l,
            classes,
        )
        return self.filter(mask)

    def view(self, cmap=[]):
        cmap = cmap if cmap != [] else self.cmap
        cpu_cld = self.to_device("cpu")
        input_cld = cpu_cld.to_o3d_cld()
        segmented_cld = o3d_cloud(cpu_cld.xyz, colours=cmap[cpu_cld.class_l])
        projected = o3d_cloud(cpu_cld.medial_pts, colour=(1, 0, 0))
        lines = o3d_lines_between_clouds(input_cld, projected)
        o3d_viewer([input_cld, projected, lines, segmented_cld])

    def to_device(self, device):
        return LabelledCloud(
            self.xyz.to(device),
            self.rgb.to(device),
            self.vector.to(device),
            self.class_l.to(device),
        )

    def cat(self):
        return torch.cat(
            (
                self.xyz,
                self.rgb,
                self.vector,
                self.class_l.unsqueeze(1),
            ),
            1,
        )

    def medial_voxel_down_sample(self, voxel_size):
        idx = voxel_downsample_idxs(self.medial_pts, voxel_size)

        return self.filter(idx)

    def to_torch(self):
        return LabelledCloud(
            torch.from_numpy(self.xyz),
            torch.from_numpy(self.rgb),
            torch.from_numpy(self.vector),
            torch.from_numpy(self.class_l),
        )

    def scale(self, factor):
        return LabelledCloud(
            self.xyz * factor, self.rgb, self.vector * factor, self.class_l
        )

    def translate(self, xyz):
        return LabelledCloud(self.xyz + xyz, self.rgb, self.vector, self.class_l)

    def rotate(self, rot_mat):
        new_xyz = torch.matmul(self.xyz, rot_mat.to(self.xyz.dtype))
        new_vectors = torch.matmul(self.vector, rot_mat.to(self.vector.dtype))

        return LabelledCloud(new_xyz, self.rgb, new_vectors, self.class_l)

    @property
    def radii(self):
        return self.vector.pow(2).sum(1).sqrt()

    @property
    def direction(self):
        return F.normalize(self.vector)

    @property
    def medial_pts(self):
        return self.xyz + self.vector

    @staticmethod
    def from_numpy(xyz, rgb, vector, class_l):
        return LabelledCloud(
            torch.from_numpy(xyz),  # float64 -> these data types are stupid...
            torch.from_numpy(rgb),  # float64
            torch.from_numpy(vector),  # float32
            torch.from_numpy(class_l),  # int64
        )
