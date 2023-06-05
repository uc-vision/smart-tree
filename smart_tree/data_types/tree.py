from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np
import torch.nn.functional as F


from ..util.math.queries import pts_to_nearest_tube_gpu
from ..util.mesh.geometries import (
    o3d_merge_clouds,
    o3d_merge_linesets,
    o3d_merge_meshes,
)
from ..util.misc import flatten_list
from .branch import BranchSkeleton
from .tube import Tube


@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]

    def __post_init__(self):
        self.colour = torch.rand(3)

    def __len__(self):
        return len(self.branches)

    def __str__(self):
        return f"Tree Skeleton ({self._id}) has {len(self)} branches..."

    def to_o3d_linesets(self) -> List:
        return [b.to_o3d_lineset() for b in self.branches.values()]

    def to_o3d_lineset(self, colour=(0, 0, 0)):
        return o3d_merge_linesets(self.to_o3d_linesets(), colour=colour)

    def to_o3d_tubes(self) -> List:
        return [b.to_o3d_tube() for b in self.branches.values()]

    def to_o3d_tube(self):
        return o3d_merge_meshes(self.to_o3d_tubes())

    def to_tubes(self) -> List[Tube]:
        return flatten_list([b.to_tubes() for b in self.branches.values()])

    def sample_skeleton(self, spacing):
        return sample_tubes(self.to_tubes(), spacing)

    def to_device(self, device):
        for branch_id, branch in self.branches():
            return BranchSkeleton(
                _id,
                parent_id,
                self.xyz.to(device),
                self.radii.to(device),
                child_id,
            )

    def repair(self):
        """skeletons are not connected between branches.
        this function connects the branches to their parent branches by finding
        the nearest point on the parent branch."""

        branch_ids = [branch._id for branch in self.branches.values()]

        for branch in self.branches.values():
            if branch.parent_id not in branch_ids:
                continue

            parent_branch = self.branches[branch.parent_id]
            tubes = parent_branch.to_tubes()

            v, idx, _ = pts_to_nearest_tube_gpu(branch.xyz[0].reshape(-1, 3), tubes)

            connection_pt = branch.xyz[0].reshape(-1, 3).cpu() + v[0].cpu()

            branch.xyz = torch.cat((connection_pt, branch.xyz))
            branch.radii = torch.cat((branch.radii[[0]], branch.radii))

    def prune(self, min_radius, min_length, root_id=0):
        """If a branch doesn't meet the initial radius threshold or length threshold we want to remove it and all
        it's predecessors...
        minimum_radius: some point of the branch must be above this to not remove the branch
        length: the total lenght of the branch must be greater than this point
        """
        branches_to_keep = {root_id: self.branches[root_id]}

        for branch_id, branch in self.branches.items():
            if branch.parent_id == -1:
                continue

            if branch.parent_id in branches_to_keep:
                if branch.length > min_length and (
                    max(branch.radii[0], branch.radii[-1])
                    > min_radius  # We don't know which end of the branch is the start for skeletons that aren't connected...
                ):
                    branches_to_keep[branch_id] = branch

        self.branches = branches_to_keep

    def smooth(self, kernel_size=5):
        """
        Smooths the skeleton radius.
        """
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        for branch in self.branches.values():
            if branch.radii.shape[0] > kernel_size:
                branch.radii = F.conv1d(
                    branch.radii.reshape(1, 1, -1),
                    kernel,
                    padding="same",
                ).reshape(-1)


@dataclass
class DisjointTreeSkeleton:
    skeletons: List[TreeSkeleton]

    def prune(self, min_radius, min_length):
        self.skeletons[0].prune(
            min_radius=min_radius,
            min_length=min_length,
        )  # Can only prune the first skeleton as we don't know the root points for all the other skeletons...

    def repair(self):
        for skeleton in self.skeletons:
            skeleton.repair()

    def smooth(self, kernel_size=10):
        for skeleton in self.skeletons:
            skeleton.smooth(kernel_size=kernel_size)

    def to_o3d_lineset(self):
        return o3d_merge_linesets(
            [s.to_o3d_lineset().paint_uniform_color(s.colour) for s in self.skeletons]
        )

    def to_o3d_tube(self, colour=True):
        if colour:
            skeleton_tubes = [
                skel.to_o3d_tube().paint_uniform_color(skel.colour)
                for skel in self.skeletons
            ]
        else:
            skeleton_tubes = [s.to_o3d_tube() for s in self.skeletons]

        return o3d_merge_meshes(skeleton_tubes)

    def connect():
        pass
