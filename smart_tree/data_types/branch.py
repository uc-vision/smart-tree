from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import open3d as o3d
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_path, o3d_tube_mesh
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from .tube import Tube

patch_typeguard()


@typechecked
@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: TensorType["N", 3, float]
    radii: TensorType["N", 1, float]

    branch_direction: Optional[TensorType["N", 3, float]] = None
    child_id: Optional[int] = None
    colour: Optional[TensorType[3]] = torch.rand(3)

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Branch ({self._id}):\n"
            f"Number Points:{self.xyz.shape[0]}\n"
            f"Length: {self.length}\n"
            f"{'*' * 80}"
        )

    def to_tubes(self) -> List[Tube]:
        return list(map(Tube, self.xyz, self.xyz[1:], self.radii, self.radii[1:]))

    def filter(self, mask: TensorType["N", torch.bool]) -> BranchSkeleton:
        args = asdict(self)
        args["xyz"] = args["xyz"][mask]
        args["radii"] = args["radii"][mask]
        if self.branch_direction is not None:
            args["branch_direction"] = args["branch_direction"][mask]
        return BranchSkeleton(**args)

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            if v is isinstance(v, torch.Tensor):
                args[k] = v.to(device)
        return BranchSkeleton(**args)

    @property
    def length(self) -> TensorType[torch.float]:
        return torch.diff(self.xyz, dim=0).norm(dim=1).sum()

    @property
    def initial_radius(self) -> TensorType[torch.float]:
        return torch.max(self.radii[0], self.radii[-1])

    def as_o3d_lineset(self) -> o3d.geometry.LineSet:
        return o3d_path(self.xyz, self.colour)

    def as_o3d_tube(self) -> o3d.geometry.TriangleMesh:
        return o3d_tube_mesh(self.xyz, self.radii, self.colour)

    @property
    def viewer_items(self) -> list[ViewerItem]:
        items = []
        items += [ViewerItem(f"Branch {self._id} Lineset", self.as_o3d_lineset())]
        items += [ViewerItem(f"Branch {self._id} Tube", self.as_o3d_tube())]
        return items

    def view(self):
        o3d_viewer(self.viewer_items)
