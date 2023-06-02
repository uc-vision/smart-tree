from dataclasses import dataclass
from typing import List

import torch
import numpy as np
import pandas as pd

from smart_tree.util.mesh.geometries import o3d_path, o3d_tube_mesh

from .tube import Tube


@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: torch.Tensor  # N x 3
    radii: torch.Tensor  # N x 1
    child_id: int = -1

    def __post_init__(self):
        self.colour = np.random.rand(3)

    def __len__(self) -> np.array:
        return self.xyz.shape[0]

    def __str__(self):
        return f" ID: {self._id} \
                  Points: {self.xyz} \
                  Radii {self.radii}"

    def to_o3d_lineset(self, colour=(0, 0, 0)):
        return o3d_path(self.xyz, colour)

    def to_o3d_tube(self):
        return o3d_tube_mesh(self.xyz.numpy(), self.radii.numpy(), self.colour)

    def to_tubes(self, colour=(1, 0, 0)) -> List[Tube]:
        a_, b_, r1_, r2_ = (
            self.xyz[:-1],
            self.xyz[1:],
            self.radii[:-1],
            self.radii[1:],
        )
        return [Tube(a, b, r1, r2) for a, b, r1, r2 in zip(a_, b_, r1_, r2_)]

    def filter(self, mask):
        return BranchSkeleton(
            self._id,
            self.parent_id,
            self.xyz[mask],
            self.radii[mask],
            self.child_id,
        )

    @property
    def length(self) -> float:
        return np.sum(np.sqrt(np.sum(np.diff(self.xyz, axis=0) ** 2, axis=1)))
