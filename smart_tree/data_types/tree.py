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
from ..util.queries import pts_to_nearest_tube
from .branch import BranchSkeleton
from .graph import Graph, join_graphs
from .line import LineSegment
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

            v, idx, _ = pts_to_nearest_tube(
                branch.xyz[[0]],
                tubes,
                device=torch.device("cuda"),
            )

            connection_pt = branch.xyz[[0]].cpu() + v[0].cpu()

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

    def branch_from_tube_idx(self, idx):
        return self.branches[int(self.find_branch_id_from_tube_idx(idx))]

    def find_branch_id_from_tube_idx(self, idx):
        number_verts = torch.tensor([len(b.to_tubes()) for b in self.branches.values()])
        number_verts = torch.cumsum(number_verts, dim=0)
        branch_id = list(self.branches.keys())[
            torch.searchsorted(number_verts, idx).cpu()
        ]
        return branch_id

    @property
    def length(self) -> TensorType:
        return torch.sum(torch.tensor([b.length for b in self.branches.values()]))

    @property
    def biggest_radius_idx(self) -> TensorType[1]:
        return torch.argmax(self.radii)

    @property
    def max_branch_id(self) -> int:
        return max(self.branches.keys())

    @property
    def xyz(self) -> TensorType["N", 3]:
        xyz = []
        for branch in self.branches.values():
            xyz.append(branch.xyz)
        if len(xyz) == 0:
            return torch.tensor(xyz)
        return torch.cat(xyz, dim=0)

    def to_tubes(self) -> List[Tube]:
        return flatten_list([b.to_tubes() for b in self.branches.values()])

    def to_line_segments(self) -> List[LineSegment]:
        return flatten_list([b.to_line_segments() for b in self.branches.values()])

    def to_graph(self) -> Graph:
        graphs = []

        offsets = {}
        offset = 0

        for branch_id, branch in self.branches.items():
            offsets[branch_id] = offset
            branch_graph = branch.to_graph()

            if branch.parent_id != -1:
                parent_branch = self.branches[branch.parent_id]
                parent_tubes = parent_branch.to_tubes()
                parent_offset = torch.tensor(offsets[branch.parent_id])

                _, idx, _ = pts_to_nearest_tube(
                    branch.xyz[[0]],
                    parent_tubes,
                    device=torch.device("cuda"),
                )

                branch_graph.edges += offsets[branch_id]

                parent_edge_idx = (idx + parent_offset).cpu()

                new_edge = torch.hstack(
                    (parent_edge_idx, torch.tensor(offset))
                ).reshape(-1, 2)

                branch_graph.edges = torch.cat((branch_graph.edges, new_edge))
                branch_graph.edge_weights = torch.cat(
                    (branch_graph.edge_weights, torch.tensor([[1]]))
                )

            graphs.append(branch_graph)
            offset += len(branch)

        return join_graphs(graphs, offset_edges=False)

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
        o3d_viewer(self.viewer_items)

    def add_branches(self, branches: dict):
        pass


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

    @property
    def viewer_items(self) -> list[ViewerItem]:
        items = []
        items += [ViewerItem(f"Skeleton Lineset", self.as_o3d_lineset())]
        items += [ViewerItem(f"Skeleton Tube", self.as_o3d_tube())]
        items += [ViewerItem(f"Skeleton Coloured Tube", self.as_o3d_tube(colour=True))]
        return items

    def view(self):
        o3d_viewer(self.viewer_items)

    def to_pickle(self, path):
        with open(f"{path}", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def from_pickle(path):
        with open(f"{path}", "rb") as pickle_file:
            return pickle.load(pickle_file)


def generate_new_branch_ids(tree: TreeSkeleton, offset: int):
    # Generate new branch IDs starting from 'offset'
    new_branch_ids = {}
    for branch_id in tree.branches.keys():
        new_branch_ids[branch_id] = branch_id + offset
    return new_branch_ids


def merge_trees(tree1: TreeSkeleton, tree2: TreeSkeleton):
    # Generate new branch IDs for tree2 branches
    max_branch_id = max(tree1.branches.keys())
    new_branch_ids = generate_new_branch_ids(tree2, max_branch_id + 1)

    # Merge tree2 branches into tree1 while updating IDs
    merged_branches = {}
    for branch_id, branch in tree1.branches.items():
        merged_branches[branch_id] = branch
    for branch_id, branch in tree2.branches.items():
        merged_branch = branch
        # Update branch ID and parent ID
        merged_branch._id = new_branch_ids[branch_id]
        merged_branch.parent_id = new_branch_ids.get(branch.parent_id, branch.parent_id)
        merged_branches[new_branch_ids[branch_id]] = merged_branch

    # Create a new TreeSkeleton instance with merged branches
    merged_tree = TreeSkeleton(
        _id=tree1._id, branches=merged_branches, colour=tree1.colour
    )
    return merged_tree


def merge_trees_in_list(tree_list):
    if not tree_list:
        return None

    merged_tree = tree_list[0]

    for tree in tree_list[1:]:
        merged_tree = merge_trees(merged_tree, tree)

    return merged_tree
