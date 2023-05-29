import json
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import yaml
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from smart_tree.data_types.branch import BranchSkeleton
from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.data_types.tree import TreeSkeleton
from smart_tree.util.mesh.geometries import o3d_cloud, o3d_line_set


def unpackage_data(data: dict) -> Tuple[LabelledCloud, TreeSkeleton]:
    tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"]
    skeleton_radii = data["skeleton_radii"]
    sizes = data["branch_num_elements"]

    cld = LabelledCloud.from_numpy(
        xyz=data["xyz"],
        rgb=data["rgb"],
        vector=data["vector"],
        class_l=data["class_l"],
    )

    return cld, None

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return cld, TreeSkeleton(tree_id, branches)


def package_data(skeleton: TreeSkeleton, pointcloud: LabelledCloud) -> dict:
    data = {}

    data["tree_id"] = skeleton._id

    data["xyz"] = pointcloud.xyz
    data["rgb"] = pointcloud.rgb
    data["vector"] = pointcloud.vector
    data["class_l"] = pointcloud.class_l

    data["skeleton_xyz"] = np.concatenate(
        [branch.xyz for branch in skeleton.branches.values()]
    )
    data["skeleton_radii"] = np.concatenate(
        [branch.radii for branch in skeleton.branches.values()]
    )[..., np.newaxis]
    data["branch_id"] = np.asarray(
        [branch._id for branch in skeleton.branches.values()]
    )
    data["branch_parent_id"] = np.asarray(
        [branch.parent_id for branch in skeleton.branches.values()]
    )
    data["branch_num_elements"] = np.asarray(
        [len(branch) for branch in skeleton.branches.values()]
    )

    return data


def save_skeleton(skeleton: TreeSkeleton, save_location):
    data = {}
    data["tree_id"] = skeleton._id

    data["skeleton_xyz"] = np.concatenate(
        [branch.xyz for branch in skeleton.branches.values()]
    )
    data["skeleton_radii"] = np.concatenate(
        [branch.radii for branch in skeleton.branches.values()]
    )[..., np.newaxis]
    data["branch_id"] = np.asarray(
        [branch._id for branch in skeleton.branches.values()]
    )
    data["branch_parent_id"] = np.asarray(
        [branch.parent_id for branch in skeleton.branches.values()]
    )
    data["branch_num_elements"] = np.asarray(
        [len(branch) for branch in skeleton.branches.values()]
    )

    np.savez(save_location, **data)


def load_skeleton(path):
    data = np.load(path)

    # tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"]
    skeleton_radii = data["skeleton_radii"]
    sizes = data["branch_num_elements"]

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return TreeSkeleton(0, branches)


def save_data_npz(path: Path, skeleton: TreeSkeleton, pointcloud: LabelledCloud):
    np.savez(path, **package_data(skeleton, pointcloud))


def load_data_npz(path: Path) -> Tuple[LabelledCloud, TreeSkeleton]:
    with np.load(path) as data:
        return unpackage_data(data)


def load_json(json_path):
    return json.load(open(json_path))


def save_o3d_cloud(filename: Path, cld: LabelledCloud):
    o3d.io.write_point_cloud(str(filename), cld)


def save_o3d_lineset(path: Path, ls):
    return o3d.io.write_line_set(path, ls)


def save_o3d_mesh(path: Path, mesh):
    return o3d.io.write_triangle_mesh(path, mesh)


def load_o3d_cloud(path: Path):
    return o3d.io.read_point_cloud(path)


def load_o3d_lineset(path: Path):
    return o3d.io.read_line_set(path)


def load_o3d_mesh(path: Path):
    return o3d.io.read_triangle_mesh(path)


def save_skeleton(filename: Path, skeleton: TreeSkeleton):
    # if filename.is_file():
    #  print("File Already Exists")
    #  return

    cld = o3d_cloud(cld.xyz)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    cld = cld.rotate(R, center=(0, 0, 0))

    o3d.io.write_point_cloud(str(filename), cld)


def load_adtree_skeleton(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.column_stack(
        (
            np.asarray(plydata.elements[0].data["x"]),
            np.asarray(plydata.elements[0].data["y"]),
            np.asarray(plydata.elements[0].data["z"]),
        )
    ).reshape(-1, 3)

    # radii = np.asarray(plydata.elements[0].data['radius'])
    edges = np.asarray(plydata.elements[1].data["vertex_indices"])

    return o3d_line_set(xyz, edges)


def load_cloud(path: Path):
    if path.suffix == ".npz":
        np_data = np.load(path)
        xyz, rgb = np_data["xyz"], np_data["rgb"]
    else:
        data = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(data.points)
        rgb = np.asarray(data.colors)

    return Cloud.from_numpy(xyz, rgb)


def load_yaml(path: Path):
    with open(f"{path}") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
