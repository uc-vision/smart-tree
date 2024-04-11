from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import open3d as o3d
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_path, o3d_tube_mesh
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from .base import Base
from .graph import Graph
from .line import LineSegment
from .tube import Tube

patch_typeguard()


@typechecked
@dataclass
class BranchSkeleton(Base):
    _id: int
    parent_id: int
    xyz: TensorType["N", 3, float]
    radii: TensorType["N", 1, float]

    branch_direction: Optional[TensorType["N", 3, float]] = None
    child_id: Optional[int] = None
    colour: TensorType[3] = field(default_factory=lambda: torch.rand(3))

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Branch ({self._id}):\n"
            f"Number Points:{len(self)}\n"
            f"Length: {self.length}\n"
            f"{'*' * 80}"
        )

    def as_tubes(self) -> List[Tube]:
        return list(map(Tube, self.xyz, self.xyz[1:], self.radii, self.radii[1:]))

    def as_line_segments(self) -> List[LineSegment]:
        return list(map(LineSegment, self.xyz, self.xyz[1:]))

    def as_graph(self) -> Graph:
        edge_idx = torch.arange(len(self.xyz), dtype=torch.long).unsqueeze(0)
        edge_idx = torch.cat([edge_idx[:, :-1], edge_idx[:, 1:]], dim=0)
        edge_weights = (self.xyz[1:] - self.xyz[:-1]).norm(dim=1)

        return Graph(
            self.xyz,
            edge_idx,
            edge_weights,
            radius=self.radii,
            branch_direction=self.branch_direction,
        )

    @property
    def length(self) -> TensorType[float]:
        return torch.diff(self.xyz, dim=0).norm(dim=1).sum()

    @property
    def initial_radius(self) -> float:
        return max(self.radii[0].item(), self.radii[-1].item())

    def as_o3d_lineset(self) -> o3d.geometry.LineSet:
        return o3d_path(self.xyz, self.colour)

    def as_o3d_tube(self) -> o3d.geometry.TriangleMesh:
        return o3d_tube_mesh(self.xyz, self.radii, self.colour)

    @property
    def viewer_items(self) -> list[ViewerItem]:
        return [
            ViewerItem(f"Branch {self._id} Lineset", self.as_o3d_lineset()),
            ViewerItem(f"Branch {self._id} Tube", self.as_o3d_tube()),
        ]

    def view(self):
        o3d_viewer(self.viewer_items)
