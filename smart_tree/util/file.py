from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import yaml

from tqdm import tqdm

from smart_tree.data_types.branch import BranchSkeleton
from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.data_types.tree import TreeSkeleton


class CloudLoader:
    def load(self, file: str | Path):
        file_path = Path(file)
        if file_path.suffix == ".npz":
            return self.load_numpy(file_path)
        else:
            return self.load_o3d(file_path)

    def load_o3d(self, file: str | Path) -> Cloud:
        try:
            pcd = o3d.io.read_point_cloud(str(file))
            return self._load_as_cloud(
                {"xyz": np.asarray(pcd.points), "rgb": np.asarray(pcd.colors)}, file
            )
        except Exception as e:
            raise ValueError(f"Error loading file {file}: {e}")

    def load_numpy(self, file: str | Path) -> Cloud | LabelledCloud:
        data = np.load(file, allow_pickle=True)
        optional_params = [
            f.name for f in fields(LabelledCloud) if f.default is not None
        ]

        for param in optional_params:
            if param in data:
                return self._load_as_labelled_cloud(data, file)

        return self._load_as_cloud(data, file)

    def _load_as_cloud(self, data, file_path: Path) -> Cloud:
        cloud_fields = self._extract_fields(data, Cloud, file_path)
        return Cloud(**cloud_fields)

    def _load_as_labelled_cloud(self, data, file_path: Path) -> LabelledCloud:
        labelled_cloud_fields = self._extract_fields(data, LabelledCloud, file_path)
        return LabelledCloud(**labelled_cloud_fields)

    def _extract_fields(self, data, cloud_type, file_path: Path) -> dict:
        fields_dict = {
            f.name: data[f.name] for f in fields(cloud_type) if f.name in data
        }
        for key, value in tqdm(fields_dict.items(), desc="Loading cloud", leave=False):
            if isinstance(value, np.ndarray):
                dtype = torch.long if key in ["class_l", "branch_ids"] else torch.float
                if value.ndim == 1:
                    fields_dict[key] = torch.from_numpy(value).type(dtype).unsqueeze(1)
                else:
                    fields_dict[key] = torch.from_numpy(value).type(dtype)

        fields_dict["filename"] = file_path

        if "vector" in data.keys():
            fields_dict["medial_vector"] = torch.from_numpy(data["vector"]).type(dtype)

        return fields_dict


# class CloudLoader:
#     def load(self, file: str | Path):
#         file_path = Path(file)
#         if file_path.suffix == ".npz":
#             return self.load_numpy(np.load(file, allow_pickle=True))
#         else:
#             return self.load_o3d(file_path)

#     def load_o3d(self, file: str | Path) -> Cloud:
#         try:
#             pcd = o3d.io.read_point_cloud(str(file))
#             return self._load_as_cloud(
#                 {"xyz": np.asarray(pcd.points), "rgb": np.asarray(pcd.colors)}, file
#             )
#         except Exception as e:
#             raise ValueError(f"Error loading file {file}: {e}")

#     def load_numpy(self, data) -> Cloud | LabelledCloud:
#         optional_params = [
#             f.name for f in fields(LabelledCloud) if f.default is not None
#         ]

#         for param in optional_params:
#             if param in data:
#                 return self._load_as_labelled_cloud(data, file)

#         return self._load_as_cloud(data, file)

#     def _load_as_cloud(self, data, file_path: Path) -> Cloud:
#         cloud_fields = self._extract_fields(data, Cloud, file_path)
#         return Cloud(**cloud_fields)

#     def _load_as_labelled_cloud(self, data, file_path: Path) -> LabelledCloud:
#         labelled_cloud_fields = self._extract_fields(data, LabelledCloud, file_path)
#         return LabelledCloud(**labelled_cloud_fields)

#     def _extract_fields(self, data, cloud_type, file_path: Path) -> dict:
#         fields_dict = {
#             f.name: data[f.name] for f in fields(cloud_type) if f.name in data
#         }
#         for key, value in tqdm(fields_dict.items(), desc="Loading cloud", leave=False):
#             if isinstance(value, np.ndarray):
#                 dtype = torch.long if key in ["class_l", "branch_ids"] else torch.float
#                 if value.ndim == 1:
#                     fields_dict[key] = torch.from_numpy(value).type(dtype).unsqueeze(1)
#                 else:
#                     fields_dict[key] = torch.from_numpy(value).type(dtype)

#         fields_dict["filename"] = file_path

#         # if "vector" in data.keys():
#         #     fields_dict["medial_vector"] = torch.from_numpy(data["vector"]).type(dtype)

#         return fields_dict


class SkeletonLoader:
    def load(self, file: str | Path):
        if Path(file).suffix == ".npz":
            return self.process_numpy(np.load(file))
        else:
            raise ValueError(f"Unsupported file type")

    def load_numpy(self, data):
        tree_id = data["tree_id"]
        branch_id = data["branch_id"]
        branch_parent_id = data["branch_parent_id"]
        skeleton_xyz = data["skeleton_xyz"]
        skeleton_radii = data["skeleton_radii"]

        sizes = data["branch_num_elements"]

        offsets = np.cumsum(np.append([0], sizes))

        branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
        branches = {}

        for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
            branches[int(_id)] = BranchSkeleton(
                _id=int(_id),
                parent_id=int(parent_id),
                xyz=torch.tensor(skeleton_xyz[idx]).float(),
                radii=torch.tensor(skeleton_radii[idx]).float(),
            )

        return TreeSkeleton(int(tree_id), branches)


class Skeleton_and_Cloud_Loader:
    def __init__(self):
        self.skeleton_loader = SkeletonLoader()
        self.cloud_loader = CloudLoader()

    def load(self, path: Path):
        np_data = np.load(path, allow_pickle=True)
        skeleton = self.skeleton_loader.load_numpy(np_data)

        pcd = self.cloud_loader._load_as_cloud(np_data, path)
        return skeleton, pcd


# def unpackage_data(data: dict) -> Tuple[Cloud, TreeSkeleton]:
#     tree_id = data["tree_id"]
#     branch_id = data["branch_id"]
#     branch_parent_id = data["branch_parent_id"]
#     skeleton_xyz = data["skeleton_xyz"]
#     skeleton_radii = data["skeleton_radii"]
#     sizes = data["branch_num_elements"]

#     cld_loader = CloudLoader()

#     # cld_loader.load(LabelledCloud(
#     #     xyz=torch.tensor(data["xyz"]),
#     #     rgb=torch.tensor(data["rgb"]),
#     #     medial_vector=data["medial_vector"],
#     #     class_l=torch.tensor(data["class_l"]).int().reshape(-1, 1),
#     #     branch_ids=torch.tensor(data["branch_ids"]).int().reshape(-1, 1),
#     # )

#     # return cld, TreeSkeleton(int(tree_id), branches)


def package_data(skeleton: TreeSkeleton, pointcloud: Cloud) -> dict:
    data = {}

    data["tree_id"] = skeleton._id

    data["xyz"] = pointcloud.xyz
    data["rgb"] = pointcloud.rgb
    data["medial_vector"] = pointcloud.medial_vector
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


# def load_data_npz(path: Path) -> Tuple[Cloud, TreeSkeleton]:
#     with np.load(path) as data:
#         return unpackage_data(data)


def load_json(json_path):
    return json.load(open(json_path))


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


""" Open3D Abstractions """


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
