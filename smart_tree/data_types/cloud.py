from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Optional

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
    xyz: TensorType["N", 3, float]
    rgb: Optional[TensorType["N", 3, float]] = None
    filename: Optional[Path] = None

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Cloud:\n"
            f"Num pts: {self.xyz.shape[0]}\n"
            f"Coloured: {hasattr(self, 'rgb')}\n"
            f"Filename: {self.filename}\n"
            f"{'*' * 80}"
        )

    def scale(
        self,
        factor: float | TensorType[1, float] | TensorType[1, 3, float],
    ) -> Cloud:
        scaled_xyz = self.xyz * factor
        return Cloud(xyz=scaled_xyz, rgb=self.rgb, filename=self.filename)

    def translate(self, translation_vector: TensorType[3, float]) -> Cloud:
        translated_xyz = self.xyz + translation_vector
        return Cloud(xyz=translated_xyz, rgb=self.rgb, filename=self.filename)

    def rotate(self, rotation_matrix: TensorType[3, 3, float]) -> Cloud:
        rotated_xyz = torch.matmul(self.xyz, rotation_matrix.T)
        return Cloud(xyz=rotated_xyz, rgb=self.rgb, filename=self.filename)

    def voxel_downsample(self, voxel_size: float | TensorType[1]) -> Cloud:
        return self.filter(voxel_filter(self.xyz, voxel_size))

    def filter(self, mask: TensorType["N"]) -> Cloud:
        filtered_rgb = self.rgb[mask] if self.rgb is not None else None
        return Cloud(xyz=self.xyz[mask], rgb=filtered_rgb, filename=self.filename)

    def to_device(self, device: torch.device) -> Cloud:
        return Cloud(
            xyz=self.xyz.to(device),
            rgb=self.rgb.to(device) if self.rgb is not None else None,
            filename=self.filename,
        )

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
    def bounding_box(self) -> tuple[TensorType[3], TensorType[3]]:
        return self.min_xyz, self.max_xyz

    @property
    def group_name(self) -> str:
        return f"{self.filename.stem}" if self.filename != None else ""

    def as_o3d_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.xyz, colours=self.rgb)

    def viewer_items(self) -> list[ViewerItem]:
        return [ViewerItem("Cloud", self.as_o3d_cld(), False, group=self.group_name)]

    def view(self) -> None:
        o3d_viewer(self.viewer_items())


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

    def scale(self, factor: float | TensorType[1]) -> LabelledCloud:
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

    def filter(self, mask: TensorType["N", torch.bool]) -> LabelledCloud:
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v[mask]
        return LabelledCloud(**args)

    def filter_by_class(self, classes: TensorType["N"]) -> LabelledCloud:
        mask = torch.isin(self.class_l, classes)
        return self.filter(mask.view(-1))

    def pin_memory(self):
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.pin_memory()
        return LabelledCloud(**args)

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.to(device)
        return LabelledCloud(**args)

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return torch.unique(self.class_l).shape[0]

    @property
    def radius(self) -> TensorType["N", 1]:
        return self.medial_vector.pow(2).sum(1).sqrt().unsqueeze(1)

    @property
    def medial_direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector)

    def as_o3d_segmented_cld(
        self,
        cmap: TensorType["N", 3] = None,
    ) -> o3d.geometry.PointCloud:
        if cmap is None:
            cmap = torch.rand(self.number_classes, 3)
        colours = cmap.to(self.device)[self.class_l.view(-1).int()]
        return o3d_cloud(self.xyz, colours=colours)

    def as_o3d_trunk_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter(self.branch_ids == trunk_id).as_o3d_cld()

    def as_o3d_branch_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter(self.branch_ids != trunk_id).as_o3d_cld()

    def as_o3d_medial_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.xyz + self.medial_vector)

    def as_o3d_medial_vectors(self) -> o3d.geometry.LineSet:
        medial_cloud = o3d_cloud(self.xyz + self.medial_vector)
        return o3d_lines_between_clouds(self.as_o3d_cld(), medial_cloud)

    def as_o3d_branch_directions(self, view_length=0.1) -> o3d.geometry.LineSet:
        branch_dir_cloud = o3d_cloud(self.xyz + (self.branch_direction * view_length))
        return o3d_lines_between_clouds(self.to_o3d_cld(), branch_dir_cloud)

    def viewer_items(self) -> list[ViewerItem]:
        items = super().viewer_items()
        item = partial(ViewerItem, is_visible=False, group=self.group_name)
        if self.medial_vector is not None:
            items += [item("Medial Vectors", self.as_o3d_medial_vectors())]
        if self.branch_direction is not None:
            items += [item("Branch Directions", self.as_o3d_branch_directions())]
        if self.branch_ids is not None:
            items += [item("Trunk", self.as_o3d_trunk_cld())]
            items += [item("Branches", self.as_o3d_branch_cld())]
        if self.class_l is not None:
            items += [item("Segmented", self.as_o3d_segmented_cld())]
        return items

    def view(self):
        o3d_viewer(self.viewer_items())


class CloudLoader:
    def load(self, npz_file: str | Path):
        data = np.load(npz_file)

        optional_params = [
            f.name for f in fields(LabelledCloud) if f.default is not None
        ]

        for param in optional_params:
            if param in data:
                return self._load_as_labelled_cloud(data, npz_file)

        return self._load_as_cloud(data, npz_file)

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
        for k, v in labelled_cloud_fields.items():
            if k in ["class_l", "branch_ids"]:
                labelled_cloud_fields[k] = torch.from_numpy(v).long().reshape(-1, 1)

            elif k in ["medial_vector"]:
                labelled_cloud_fields["medial_vector"] = torch.from_numpy(v).float()

            else:
                labelled_cloud_fields[k] = torch.from_numpy(v).float()
        labelled_cloud_fields["filename"] = Path(fn)
        return LabelledCloud(**labelled_cloud_fields)
