import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import yaml

from smart_tree.data_types.branch import BranchSkeleton
from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.data_types.tree import TreeSkeleton


def unpackage_data(data: dict) -> Tuple[Cloud, TreeSkeleton]:
    tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"]
    skeleton_radii = data["skeleton_radii"]
    sizes = data["branch_num_elements"]

    cld = LabelledCloud(
        xyz=torch.tensor(data["xyz"]),
        rgb=torch.tensor(data["rgb"]),
        medial_vector=data["medial_vector"],
        class_l=torch.tensor(data["class_l"]).int().reshape(-1, 1),
        branch_ids=torch.tensor(data["branch_ids"]).int().reshape(-1, 1),
    )
    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[int(_id)] = BranchSkeleton(
            int(_id),
            int(parent_id),
            torch.tensor(skeleton_xyz[idx]).float(),
            torch.tensor(skeleton_radii[idx]).float(),
        )

    return cld, TreeSkeleton(int(tree_id), branches)


def package_data(skeleton: TreeSkeleton, pointcloud: Cloud) -> dict:
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


def save_cloud(cloud: Cloud | LabelledCloud, save_path):
    cloud_dict = asdict(cloud)
    np_dict = {}
    for key, value in cloud_dict.items():
        if isinstance(value, np.ndarray):
            np_dict[key] = value
        elif isinstance(value, torch.Tensor):
            np_dict[key] = value.numpy()
        # elif isinstance(value, Path):
        #     np_dict[key] = str(value)
        # else:
        #     np_dict[key] = value

    np.savez(save_path, **np_dict)


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


def save_data_npz(path: Path, skeleton: TreeSkeleton, pointcloud: Cloud):
    np.savez(path, **package_data(skeleton, pointcloud))


def load_data_npz(path: Path) -> Tuple[Cloud, TreeSkeleton]:
    with np.load(path) as data:
        return unpackage_data(data)


def load_json(json_path):
    return json.load(open(json_path))


def save_o3d_cloud(filename: Path, cld: Cloud):
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
    return o3d.io.read_triangle_model(path)


def load_cloud(path: Path):
    if path.suffix == ".npz":
        return Cloud(**np.load(path))

    data = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(data.points)
    rgb = np.asarray(data.colors) if np.asarray(data.colors).shape[0] != 0 else None

    if rgb.all() != None:
        return Cloud.from_numpy(xyz=xyz, rgb=rgb)

    return Cloud.from_numpy(xyz=xyz, rgb=np.ones((xyz.shape[0], 3)))


def load_yaml(path: Path):
    with open(f"{path}") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class CloudLoader:
    def load(self, file: str | Path):
        if Path(file).suffix == ".npz":
            return self.load_numpy(file)

        else:
            return self.load_o3d(file)

    def load_o3d(self, file: str):
        try:
            pcd = o3d.io.read_point_cloud(filename=str(file))

            return self._load_as_cloud(
                {"xyz": np.asarray(pcd.points), "rgb": np.asarray(pcd.colors)}, file
            )
        except:
            raise ValueError(f"File type {Path(file).suffix} not supported")

    def load_numpy(self, file: str | Path):
        data = np.load(file)

        optional_params = [
            f.name for f in fields(LabelledCloud) if f.default is not None
        ]

        for param in optional_params:
            if param in data:
                return self._load_as_labelled_cloud(data, file)

        return self._load_as_cloud(data, file)

    def _load_as_cloud(self, data, fn) -> Cloud:
        cloud_fields = {f.name: data[f.name] for f in fields(Cloud) if f.name in data}

        for k, v in cloud_fields.items():
            cloud_fields[k] = torch.from_numpy(v).float()
        cloud_fields["filename"] = Path(fn)
        return Cloud(**cloud_fields)

    def _load_as_labelled_cloud(self, data, fn) -> LabelledCloud:
        labelled_cloud_fields = {
            f.name: data[f.name] for f in fields(LabelledCloud) if f.name in data
        }
        cloud_data = {}
        for k, v in labelled_cloud_fields.items():
            if type(v) == np.ndarray:
                if v.any() == None:
                    continue
            if k in ["class_l", "branch_ids"]:
                cloud_data[k] = torch.from_numpy(v).long().reshape(-1, 1)
            elif k in ["medial_vector", "vector"]:
                cloud_data["medial_vector"] = torch.from_numpy(v).float()
            else:
                cloud_data[k] = torch.from_numpy(v).float()

        if "vector" in data.keys():
            cloud_data["medial_vector"] = torch.from_numpy(data["vector"]).float()

        cloud_data["filename"] = Path(fn)

        return LabelledCloud(**cloud_data)
