import os
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from ..data_types.branch import BranchSkeleton
from ..util.mesh.geometries import o3d_cloud, o3d_path, o3d_sphere, o3d_tube_mesh
from ..util.misc import flatten_list
from .graph import nn
from ..util.visualizer.view import o3d_viewer


def trace_route(preds, idx, allocated):
    cpu_preds = preds.cpu().numpy()
    path = []

    if len(allocated) > 0:
        endpoints = set(torch.cat(allocated).cpu().numpy())
    else:
        endpoints = set(allocated)

    while idx >= 0 and idx not in endpoints:
        path.append(idx)
        idx = cpu_preds[idx]

    return preds.new_tensor(path, dtype=torch.long).flip(0), idx


def select_path_points(
    points: torch.tensor, path_verts: torch.tensor, radii: torch.tensor
):
    """
    Finds points nearest to a path (specified by points with radii).
    points: (N, 3) 3d points of point cloud
    path_verts: (M, 3) 3d points of path
    radii: (M, 1) radii of path
    returns: (X, 2) index tuples of (path, point) for X points falling within the path, ordered by path index
    """

    point_path, dists, _ = nn(
        points, path_verts, r=radii.max().item()
    )  # nearest path idx for each point
    valid = dists[point_path >= 0] < radii[point_path[point_path >= 0]].squeeze(
        1
    )  # where the path idx is less than the distance to the point

    on_path = point_path.new_zeros(point_path.shape, dtype=torch.bool)
    on_path[point_path >= 0] = valid  # points that are on the path.

    idx_point = on_path.nonzero().squeeze(1)
    idx_path = point_path[idx_point]

    order = torch.argsort(idx_path)
    return idx_point[order], idx_path[order]


def find_branch_parent(idx_pt, idx_lookup):
    """Finds which branch pt is from ...
    so we can work out the branch parent..."""
    for _id, _idxs in idx_lookup.items():
        if idx_pt in _idxs:
            return int(_id)
    return -1


def sample_tree(
    medial_pts,
    medial_radii,
    preds,
    distances,
    all_points,
    root_idx=0,
    visualize=False,
):
    """
    Medial Points: NN estimated medial points
    Medial Radii: NN estimated radii of points
    Preds: Predecessor of each medial point (On path to root node)
    Distance: Distance from root node to medial points
    Surface Points: The point the medial pts got projected from..
    """

    selection_mask = preds > 0
    distances[~selection_mask] = -1  # Set their distances to negative 1...

    allocated_path_points = []

    branch_id = 0

    idx_lookup = {}
    branches = {}

    tubes = []

    while True:
        os.system("clear")
        print(f"{(distances > 0).sum().item() / medial_pts.shape[0]:.4f}")

        farthest = distances.argmax().item()  # Get fartherest away medial point

        if distances[farthest] <= 0:
            break

        path_vertices_idx, first_idx = trace_route(
            preds, farthest, allocated=allocated_path_points
        )  # Gets IDXs along a path and the first IDX of that path
        idx_points, idx_path = select_path_points(
            medial_pts,
            medial_pts[path_vertices_idx],
            medial_radii[path_vertices_idx],
        )

        distances[idx_points] = -1
        distances[idx_path] = -1

        # allocated_path_points.append(idx_path)
        allocated_path_points.append(idx_points)

        if len(path_vertices_idx) > 1:
            branches[branch_id] = BranchSkeleton(
                branch_id,
                xyz=medial_pts[path_vertices_idx].cpu().numpy(),
                radii=medial_radii[path_vertices_idx].cpu().numpy(),
                parent_id=find_branch_parent(int(first_idx), idx_lookup),
                child_id=-1,
            )

            idx_lookup[branch_id] = (
                path_vertices_idx.cpu().tolist() + idx_points.cpu().tolist()
            )
            branch_id += 1

            # tubes.append(
            #     o3d_tube_mesh(
            #         medial_pts[path_vertices_idx].cpu().numpy(),
            #         medial_radii[path_vertices_idx].cpu().numpy(),
            #     )
            # )

            # o3d_viewer(tubes)

    return branches
