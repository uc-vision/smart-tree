from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_path, o3d_tube_mesh
from .tube import Tube

patch_typeguard()


@typechecked
@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: TensorType["N", 3]
    radii: TensorType["N", 1]

    branch_direction: Optional[TensorType["N", 3]] = None
    child_id: Optional[int] = None
    colour: Optional[TensorType[3]] = torch.rand(3)

    def __len__(self) -> np.array:
        return self.xyz.shape[0]

    def __str__(self) -> str:
        return f" Branch {self._id} with {self.__len__} points"

    def to_tubes(self) -> List[Tube]:
        return list(map(Tube, self.xyz, self.xyz[1:], self.radii, self.radii[1:]))

    def filter(self, mask: TensorType["N"]) -> BranchSkeleton:
        return BranchSkeleton(
            _id=self._id,
            parent_id=self.parent_id,
            xyz=self.xyz[mask],
            radii=self.radii[mask],
            child_id=self.child_id,
        )

    def to_o3d_lineset(self) -> o3d.geometry.LineSet:
        return o3d_path(self.xyz, self.colour)

    def to_o3d_tube(self) -> o3d.geometry.TriangleMesh:
        return o3d_tube_mesh(self.xyz, self.radii, self.colour)

    @property
    def length(self) -> TensorType[1]:
        return (self.xyz[1:] - self.xyz[:-1]).norm(dim=1).sum()

    @property
    def initial_radius(self) -> TensorType[1]:
        return torch.max(self.radii[0], self.radii[-1])

    @property
    def biggest_radius_idx(self) -> TensorType[1]:
        return torch.argmax(self.radii)

    @property
    def biggest_radius(self) -> TensorType[1]:
        return torch.max(self.radii)
