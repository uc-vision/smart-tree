from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import open3d as o3d
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_merge_linesets, o3d_merge_meshes
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from ..util.misc import flatten_list
from ..util.queries import pts_to_nearest_tube_gpu
from .branch import BranchSkeleton
from .tube import Tube


@typechecked
@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]
    colour: Optional[TensorType[3]] = torch.rand(3)

    def __len__(self) -> int:
        return len(self.branches)

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Tree Skeleton {self._id} :\n"
            f"Num Branches: {len(self.branches)}\n"
            f"{'*' * 80}"
        )

    def repair(self) -> None:
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

    def prune(self, min_radius: float, min_length: float, root_id=None) -> TreeSkeleton:
        """
        If a branch doesn't meet the initial radius threshold or length threshold we want to remove it and all
        it's predecessors...
        minimum_radius: some point of the branch must be above this to not remove the branch
        length: the total lenght of the branch must be greater than this point
        """
        root_id = min(self.branches.keys()) if root_id == None else root_id

        keep = {root_id: self.branches[root_id]}
        remove = {}

        for branch_id, branch in self.branches.items():
            if branch.parent_id not in keep and branch._id != root_id:
                remove[branch_id] = branch
            elif branch.length < min_length:
                remove[branch_id] = branch
            elif branch.initial_radius < min_radius:
                remove[branch_id] = branch
            else:
                keep[branch_id] = branch

        self.branches = keep
        return TreeSkeleton(0, remove)

    def smooth(self, kernel_size: int = 5) -> None:
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
                ).view(-1)

    @property
    def length(self) -> TensorType[1]:
        return torch.sum(torch.cat([b.length for b in self.branches.values()]))

    @property
    def biggest_radius_idx(self) -> TensorType[1]:
        return torch.argmax(self.radii)

    @property
    def max_branch_id(self) -> int:
        return max(self.branches.keys())

    def to_tubes(self) -> List[Tube]:
        return flatten_list([b.to_tubes() for b in self.branches.values()])

    def to_device(self, device: torch.device):
        new_branches = {}
        for branch_id, branch in self.branches():
            new_branches[branch_id] = branch.to_device(device)

        return BranchSkeleton(self._id, new_branches, self.colour)

    def as_o3d_linesets(self) -> List[o3d.geometry.LineSet]:
        return [b.as_o3d_lineset() for b in self.branches.values()]

    def as_o3d_lineset(self) -> o3d.geometry.LineSet:
        return o3d_merge_linesets(self.as_o3d_linesets(), colour=self.colour)

    def as_o3d_tubes(self) -> List[o3d.geometry.TriangleMesh]:
        return [b.as_o3d_tube() for b in self.branches.values()]

    def as_o3d_tube(self, colour=None) -> o3d.geometry.TriangleMesh:
        return o3d_merge_meshes(self.as_o3d_tubes(), colour)

    @property
    def viewer_items(self) -> List[ViewerItem]:
        items = []
        items += [ViewerItem(f"Tree {self._id} Lineset", self.as_o3d_lineset())]
        items += [ViewerItem(f"Tree {self._id} Tube", self.as_o3d_tube())]
        return items

    def view(self):
        o3d_viewer(self.view_items)


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

    def smooth(self, kernel_size=7):
        for skeleton in self.skeletons:
            skeleton.smooth(kernel_size=kernel_size)

    def as_o3d_lineset(self):
        return o3d_merge_linesets(
            [s.as_o3d_lineset().paint_uniform_color(s.colour) for s in self.skeletons]
        )

    def as_o3d_tube(self, colour=True) -> o3d.geometry.TriangleMesh:
        if colour:
            skeleton_tubes = [
                skel.as_o3d_tube(colour=skel.colour) for skel in self.skeletons
            ]
        else:
            skeleton_tubes = [s.as_o3d_tube() for s in self.skeletons]

        return o3d_merge_meshes(skeleton_tubes)

    def viewer_items(self) -> list[ViewerItem]:
        items = []
        items += [ViewerItem(f"Skeleton Lineset", self.as_o3d_lineset())]
        items += [ViewerItem(f"Skeleton Tube", self.as_o3d_tube())]
        items += [ViewerItem(f"Skeleton Coloured Tube", self.as_o3d_tube(colour=True))]
        return items

    def view(self):
        o3d_viewer(self.viewer_items())

    def to_pickle(self, path):
        with open(f"{path}", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def from_pickle(path):
        with open(f"{path}", "rb") as pickle_file:
            return pickle.load(pickle_file)


# def connect(
#     skeleton_1: TreeSkeleton,
#     skeleton_1_parent_branch_key: int,
#     skeleton_1_parent_vert_idx: int,
#     skeleton_2: TreeSkeleton,
#     skeleton_2_child_branch_key: int,
#     skeleton_2_child_vert_idx: int,
# ) -> TreeSkeleton:
#     # This is bundy, only visually gives appearance of connection...
#     # Need to do some more processing to actually connect the skeletons...

#     parent_branch = skeleton_1.branches[skeleton_1_parent_branch_key]
#     child_branch = skeleton_2.branches[skeleton_2_child_branch_key]

#     child_branch.parent_id = skeleton_1_parent_branch_key
#     connection_pt = parent_branch.xyz[skeleton_1_parent_vert_idx]

#     child_branch.xyz = torch.cat((connection_pt.unsqueeze(0), child_branch.xyz))
#     child_branch.radii = torch.cat((child_branch.radii[[0]], child_branch.radii))

#     for key, branch in skeleton_2.branches.items():
#         branch._id += skeleton_1.max_branch_id

#         if branch.parent_id != -1:
#             branch.parent_id += skeleton_1.max_branch_id

#     return TreeSkeleton(0, merge_dictionaries(skeleton_1.branches, skeleton_2.branches))
