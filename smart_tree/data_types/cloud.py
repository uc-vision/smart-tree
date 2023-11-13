from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from ..util.misc import voxel_filter

patch_typeguard()


@typechecked
@dataclass
class Cloud:
    xyz: TensorType["N", 3]
    rgb: Optional[TensorType["N", 3]] = None
    filename: Optional[Path] = None

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}\n"
            f"Cloud:\n"
            f"Num pts: {self.xyz.shape[0]}\n"
            f"Coloured: {hasattr(self, 'rgb')}\n"
            f"Filename: {self.filename}\n"
            f"{'*' * 80}"
        )

    def scale(
        self,
        factor,
    ) -> Cloud:
        scaled_xyz = self.xyz * factor
        return Cloud(xyz=scaled_xyz, rgb=self.rgb, filename=self.filename)

    def translate(self, translation_vector: TensorType[3]) -> Cloud:
        translated_xyz = self.xyz + translation_vector
        return Cloud(xyz=translated_xyz, rgb=self.rgb, filename=self.filename)

    def rotate(self, rotation_matrix: TensorType[3, 3, float]) -> Cloud:
        rotated_xyz = torch.matmul(self.xyz, rotation_matrix.T.to(self.xyz.device))
        return Cloud(xyz=rotated_xyz, rgb=self.rgb, filename=self.filename)

    def voxel_downsample(self, voxel_size: float | TensorType[1]) -> Cloud:
        return self.filter(voxel_filter(self.xyz, voxel_size).reshape(-1))

    def filter(
        self,
        mask: TensorType["N", torch.bool] | TensorType["N", torch.int32] | torch.tensor,
    ) -> Cloud:
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v[mask]
        return self.__class__(**args)

    def pin_memory(self) -> Cloud:
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.pin_memory()
        return self.__class__(**args)

    def to_device(self, device: torch.device) -> Cloud:
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.to(device)
        return self.__class__(**args)

    def delete(self, delete_idx) -> Cloud:
        return self.filter(
            (
                torch.arange(self.xyz.shape[0], device=self.device)
                != delete_idx.to(self.device)
            ).reshape(-1)
        )

    def with_xyz(self, new_xyz):
        return Cloud(new_xyz, self.rgb, self.filename)

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    @property
    def max_xyz(self) -> TensorType[3]:
        return torch.max(self.xyz, dim=0)[0]

    @property
    def min_xyz(self) -> TensorType[3]:
        return torch.min(self.xyz, dim=0)[0]

    @property
    def centre(self) -> TensorType[3]:
        return torch.mean(self.xyz, dim=0)

    @property
    def max_y(self) -> TensorType:
        return self.max_xyz[2]

    @property
    def bounding_box(self) -> tuple[TensorType[3], TensorType[3]]:
        return self.min_xyz, self.max_xyz

    @property
    def root_idx(self) -> int:
        return torch.argmin(self.xyz[:, 1]).item()

    @property
    def group_name(self) -> str:
        return f"{self.filename.stem}" if self.filename != None else ""

    def as_o3d_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.xyz, colours=self.rgb)

    @property
    def viewer_items(self) -> list[ViewerItem]:
        return [
            ViewerItem(
                f"Cloud_{self.filename}",
                self.as_o3d_cld(),
                False,
                group=self.group_name,
            )
        ]

    def view(self) -> None:
        o3d_viewer(self.viewer_items)


@typechecked
@dataclass
class LabelledCloud(Cloud):
    """
    medial_vector: Vector towards medial axis
    branch_direction: Unit vector of branch direction of closest branch
    branch_ids: ID of closest branch
    class_l: Class of point (I.E. 0 = trunk, 1 = branch, 2 = leaf)
    loss_mask: Areas we don't want to compute any loss on i.e. near edges of a block
    vector_loss_mask: Where we don't want to compute loss for the medial vector or branch direction i.e. leaves
    """

    medial_vector: Optional[TensorType["N", 3]] = None
    branch_direction: Optional[TensorType["N", 3]] = None
    branch_ids: Optional[TensorType["N", 1]] = None
    class_l: Optional[TensorType["N", 1]] = None

    loss_mask: Optional[TensorType["N", 1]] = None
    vector_loss_mask: Optional[TensorType["N", 1]] = None

    vector: Optional[TensorType["N", 3]] = None  # Legacy

    def __str__(self):
        base_str = super().__str__()
        args = asdict(self)
        for k, v in args.items():
            if v is not None:
                base_str += f"\nContains: {k}"
        base_str += f"\n{'*' * 80}"
        return f"{base_str}"

    def translate(self, translation_vector: TensorType[3]) -> Cloud:
        args = asdict(self)
        args["xyz"] = args["xyz"] + translation_vector
        args["medial_vector"] = args["medial_vector"] + translation_vector
        return LabelledCloud(**args)

    def scale(self, factor: float | TensorType[1] | torch.tensor) -> LabelledCloud:
        args = asdict(self)
        args["xyz"] = args["xyz"] * factor
        if args["medial_vector"] is not None:
            args["medial_vector"] = args["medial_vector"] * factor
        return LabelledCloud(**args)

    def rotate(self, rot_matrix: TensorType[3, 3]) -> LabelledCloud:
        args = asdict(self)
        args["xyz"] = torch.matmul(args["xyz"], rot_matrix.T)
        if args["medial_vector"] is not None:
            args["medial_vector"] = torch.matmul(args["medial_vector"], rot_matrix.T)
        if args["branch_direction"] is not None:
            args["branch_direction"] = torch.matmul(
                args["branch_direction"],
                rot_matrix.T,
            )
        return LabelledCloud(**args)

    def filter_by_class(self, classes: TensorType["N"]) -> LabelledCloud:
        mask = torch.isin(self.class_l, classes)
        return self.filter(mask.view(-1))

    def filter(self, mask: TensorType) -> LabelledCloud:
        return super().filter(mask)

    def with_xyz(self, new_xyz):
        new_cld = deepcopy(self)
        new_cld.xyz = new_xyz
        return new_cld

    def pin_memory(self) -> LabelledCloud:
        return super().pin_memory()

    def to_device(self, device: torch.device) -> Cloud:
        return super().to_device(device)

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return torch.max(self.class_l).item() - torch.min(self.class_l).item() + 1

    @property
    def radius(self) -> TensorType["N", 1]:
        return self.medial_vector.pow(2).sum(1).sqrt().unsqueeze(1)

    @property
    def medial_direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector)

    @property
    def medial_pts(self) -> TensorType["N", 3]:
        return self.xyz + self.medial_vector

    def as_o3d_segmented_cld(
        self,
        cmap: TensorType["N", 3] = None,
    ) -> o3d.geometry.PointCloud:
        if cmap is None:
            cmap = torch.rand(self.number_classes, 3)

        colours = cmap.to(self.device)[self.class_l.view(-1).long()]
        return o3d_cloud(self.xyz, colours=colours)

    def as_o3d_trunk_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter((self.branch_ids == trunk_id).view(-1)).as_o3d_cld()

    def as_o3d_branch_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter((self.branch_ids != trunk_id).view(-1)).as_o3d_cld()

    def as_o3d_loss_mask_cld(self) -> o3d.geometry.PointCloud:
        cmap = torch.tensor([[1, 0, 0], [0, 1, 0]])
        colours = cmap.to(self.device)[self.loss_mask.view(-1).int()]
        return o3d_cloud(self.xyz, colours=colours)

    def as_o3d_medial_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.medial_pts)

    def as_o3d_medial_vectors(self) -> o3d.geometry.LineSet:
        return o3d_lines_between_clouds(self.as_o3d_cld(), self.as_o3d_medial_cld())

    def as_o3d_branch_directions(self, view_length=0.1) -> o3d.geometry.LineSet:
        branch_dir_cloud = o3d_cloud(self.xyz + (self.branch_direction * view_length))
        return o3d_lines_between_clouds(self.as_o3d_cld(), branch_dir_cloud)

    @property
    def viewer_items(self) -> list[ViewerItem]:
        items = super().viewer_items
        item = partial(ViewerItem, is_visible=False, group=f"{super().group_name}")
        if self.medial_vector is not None:
            items += [item("Medial Vectors", self.as_o3d_medial_vectors())]
            items += [item("Medial Cloud", self.as_o3d_medial_cld())]
        if self.branch_direction is not None:
            items += [item("Branch Directions", self.as_o3d_branch_directions())]
        if self.branch_ids is not None:
            items += [item("Trunk", self.as_o3d_trunk_cld())]
            items += [item("Branches", self.as_o3d_branch_cld())]
        if self.class_l is not None:
            items += [item("Segmented", self.as_o3d_segmented_cld())]
        if self.loss_mask is not None:
            items += [item("Loss Mask", self.as_o3d_loss_mask_cld())]
        return items

    def view(self):
        o3d_viewer(self.viewer_items)


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

        cloud_data["filename"] = Path(fn)

        return LabelledCloud(**cloud_data)


def convert_cloud_to_labelled_cloud(
    cloud: Cloud,
    medial_vector: Optional[TensorType["N", 3]] = None,
    branch_direction: Optional[TensorType["N", 3]] = None,
    branch_ids: Optional[TensorType["N", 1]] = None,
    class_l: Optional[TensorType["N", 1]] = None,
    loss_mask: Optional[TensorType["N", 1]] = None,
    vector_loss_mask: Optional[TensorType["N", 1]] = None,
    vector: Optional[TensorType["N", 3]] = None,
) -> LabelledCloud:
    cloud_dict = asdict(cloud)

    cloud_dict["medial_vector"] = medial_vector
    cloud_dict["branch_direction"] = branch_direction
    cloud_dict["branch_ids"] = branch_ids
    cloud_dict["class_l"] = class_l
    cloud_dict["loss_mask"] = loss_mask
    cloud_dict["vector_loss_mask"] = vector_loss_mask
    cloud_dict["vector"] = vector

    labelled_cloud = LabelledCloud(**cloud_dict)

    return labelled_cloud


def merge_labelled_cloud(clouds: List["LabelledCloud"]) -> "LabelledCloud":
    """Merge a list of LabelledCloud instances into a single LabelledCloud."""
    all_xyz, all_rgb, all_medial_vector, all_branch_direction = [], [], [], []
    all_branch_ids, all_class_l, all_loss_mask, all_vector_loss_mask, all_vector = (
        [],
        [],
        [],
        [],
        [],
    )

    def accumulate_tensors(accumulated_list, tensor):
        if tensor is not None:
            accumulated_list.append(tensor)

    for cloud in clouds:
        accumulate_tensors(all_xyz, cloud.xyz)
        accumulate_tensors(all_rgb, cloud.rgb)
        accumulate_tensors(all_medial_vector, cloud.medial_vector)
        accumulate_tensors(all_branch_direction, cloud.branch_direction)
        accumulate_tensors(all_branch_ids, cloud.branch_ids)
        accumulate_tensors(all_class_l, cloud.class_l)
        accumulate_tensors(all_loss_mask, cloud.loss_mask)
        accumulate_tensors(all_vector_loss_mask, cloud.vector_loss_mask)
        accumulate_tensors(all_vector, cloud.vector)  # Legacy

    # Concatenate all accumulated tensors
    new_xyz = torch.cat(all_xyz, dim=0) if all_xyz else None
    new_rgb = torch.cat(all_rgb, dim=0) if all_rgb else None

    new_medial_vector = (
        torch.cat(all_medial_vector, dim=0) if all_medial_vector else None
    )
    new_branch_direction = (
        torch.cat(all_branch_direction, dim=0) if all_branch_direction else None
    )
    new_branch_ids = torch.cat(all_branch_ids, dim=0) if all_branch_ids else None
    new_class_l = torch.cat(all_class_l, dim=0) if all_class_l else None
    new_loss_mask = torch.cat(all_loss_mask, dim=0) if all_loss_mask else None
    new_vector_loss_mask = (
        torch.cat(all_vector_loss_mask, dim=0) if all_vector_loss_mask else None
    )
    new_vector = torch.cat(all_vector, dim=0) if all_vector else None  # Legacy

    # Return a new LabelledCloud with the concatenated tensors
    return LabelledCloud(
        xyz=new_xyz,
        rgb=new_rgb,
        medial_vector=new_medial_vector,
        branch_direction=new_branch_direction,
        branch_ids=new_branch_ids,
        class_l=new_class_l,
        loss_mask=new_loss_mask,
        vector_loss_mask=new_vector_loss_mask,
        vector=new_vector,  # Legacy
    )
