from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import List, Optional

import open3d as o3d
import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from .base import Base

patch_typeguard()


@typechecked
@dataclass
class Cloud(Base):
    xyz: TensorType["N", 3, torch.float32]
    rgb: Optional[TensorType["N", 3]] = None
    filename: Optional[Path] = None

    _root_idx: Optional[int] = None

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

    def scale(self, factor) -> Cloud:
        scaled_xyz = self.xyz * factor
        return self.__class__(xyz=scaled_xyz, rgb=self.rgb, filename=self.filename)

    def translate(self, translation_vector: TensorType[3]) -> Cloud:
        translated_xyz = self.xyz + translation_vector
        return self.__class__(xyz=translated_xyz, rgb=self.rgb, filename=self.filename)

    def rotate(self, rotation_matrix: TensorType[3, 3, float]) -> Cloud:
        rotated_xyz = torch.matmul(self.xyz, rotation_matrix.T.to(self.xyz.device)).to(
            torch.float32
        )

        return self.__class__(xyz=rotated_xyz, rgb=self.rgb, filename=self.filename)

    def delete(self, delete_idx) -> Cloud:
        return self.filter(
            (
                torch.arange(self.xyz.shape[0], device=self.device)
                != delete_idx.to(self.device)
            ).reshape(-1)
        )

    def to_labelled_cloud(self, **kwargs) -> LabelledCloud:
        args = asdict(self)
        args.update(kwargs)
        return LabelledCloud(**args)

    def with_xyz(self, new_xyz):
        args = asdict(self)
        args["xyz"] = new_xyz
        return self.__class__(**args)

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
    def bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        # defined by centre coordinate, x/2, y/2, z/2
        dimensions = (self.max_xyz - self.min_xyz) / 2
        centre = self.min_xyz + dimensions
        return centre, dimensions

    @property
    def centre(self) -> TensorType[3]:
        return torch.mean(self.xyz, dim=0)

    @property
    def bounding_box(self) -> tuple[TensorType[3], TensorType[3]]:
        return self.min_xyz, self.max_xyz

    @property
    def root_idx(self) -> int | None:
        return (
            torch.argmin(self.xyz[:, 2]).item()
            if self._root_idx == None
            else int(self._root_idx)
        )

    @property
    def group_name(self) -> str:
        return f"{self.filename.stem}" if self.filename != None else ""

    def as_o3d_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.xyz, colours=self.rgb)

    def as_voxel_grid(self, voxel_size=0.0025):
        return o3d.geometry.VoxelGrid.create_from_point_cloud(
            self.as_o3d_cld(), voxel_size=voxel_size
        )

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
    A class representing a point cloud with additional labeled data.
    """

    vector: Optional[TensorType["N", 3]] = None

    medial_vector: Optional[TensorType["N", 3]] = None
    """ Vector towards medial axis. """

    branch_ids: Optional[TensorType["N", 1]] = None
    """ ID of the closest branch. """

    class_l: Optional[torch.Tensor] = None
    """ Class of point (e.g., 0 = trunk, 1 = branch, 2 = leaf). """

    point_ids: Optional[TensorType["N", 2]] = None
    """ Batch number and point number. """

    class_loss_mask: Optional[TensorType["N", 1, torch.bool]] = None
    """ Mask for areas where class loss is not computed (e.g., near edges). """

    vector_loss_mask: Optional[TensorType["N", 1, torch.bool]] = None
    """ Mask for areas where vector loss is not computed (e.g., leaves). """

    loss_mask: Optional[TensorType["N", 1, torch.bool]] = None

    def __post_init__(self):
        mask_shape = (len(self.xyz), 1)

        if self.class_loss_mask is None:
            self.class_loss_mask = torch.ones(
                mask_shape, dtype=torch.bool, device=self.device
            )

        if self.vector_loss_mask is None:
            self.vector_loss_mask = torch.ones(
                mask_shape, dtype=torch.bool, device=self.device
            )

        if self.point_ids is None:
            self.point_ids = torch.arange(len(self.xyz)).unsqueeze(1)
            zeros_column = torch.zeros_like(self.point_ids)
            self.point_ids = torch.cat((zeros_column, self.point_ids), dim=1).to(
                self.device
            )

    def __str__(self):
        base_str = super().__str__()
        args = asdict(self)
        for k, v in args.items():
            if v is not None:
                base_str += f"\nContains: {k}"
                if isinstance(v, torch.Tensor):
                    base_str += f" Shape: {tuple(v.shape)}"
        base_str += f"\n{'*' * 80}"
        return f"{base_str}"

    def translate(self, translation_vector: TensorType[3]) -> Cloud:
        args = asdict(self)
        args["xyz"] = args["xyz"] + translation_vector
        return LabelledCloud(**args)

    def scale(self, factor: float | TensorType[1] | torch.tensor) -> LabelledCloud:
        args = asdict(self)
        args["xyz"] = args["xyz"] * factor
        if args["medial_vector"] is not None:
            args["medial_vector"] = args["medial_vector"] * factor
        return LabelledCloud(**args)

    def rotate(self, rot_matrix: TensorType[3, 3]) -> LabelledCloud:
        args = asdict(self)
        xyz_rot_mat = rot_matrix.T.to(self.device).to(self.xyz.dtype)

        args["xyz"] = torch.matmul(args["xyz"], xyz_rot_mat).to(self.xyz.dtype)
        if args["medial_vector"] is not None:
            medial_rot_mat = rot_matrix.T.to(self.device).to(self.medial_vector.dtype)
            args["medial_vector"] = torch.matmul(args["medial_vector"], medial_rot_mat)

        return LabelledCloud(**args)

    def filter_by_class(self, classes: TensorType["N"]) -> LabelledCloud:
        mask = torch.isin(self.class_l, classes)
        return self.filter(mask.view(-1))

    def add_xyz(self, xyz, class_l=None):
        self.xyz = torch.cat((self.xyz, xyz))
        padd_vector = torch.zeros(
            (xyz.shape[0], 3),
            device=self.device,
            dtype=self.xyz.dtype,
        )

        self.medial_vector = torch.cat((self.medial_vector, padd_vector))
        if class_l != None:
            self.class_l = torch.cat((self.class_l, class_l))

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return int(torch.max(self.class_l).item() - torch.min(self.class_l).item()) + 1

    @property
    def radius(self) -> TensorType["N", 1]:
        return self.medial_vector.pow(2).sum(1).sqrt().unsqueeze(1)

    @property
    def medial_direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector.float())

    @property
    def medial_pts(self) -> TensorType["N", 3]:
        return self.xyz + self.medial_vector

    @property
    def has_class_labels(self) -> bool:
        return False if self.class_l == None else True

    def as_o3d_segmented_cld(
        self,
        cmap: TensorType["N", 3] = None,
    ) -> o3d.geometry.PointCloud:
        if cmap is None:
            cmap = torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # Red
                    [0.0, 1.0, 0.0],  # Green
                    [0.0, 0.0, 1.0],  # Blue
                    [1.0, 1.0, 0.0],  # Yellow
                    [0.0, 1.0, 1.0],  # Cyan
                    [1.0, 0.5, 1.0],  # Pink
                    [1.0, 0.8, 0.1],  # Orange
                    [0.5, 0.0, 0.5],  # Purple
                    [0.5, 0.5, 0.0],  # Olive
                    [0.8, 0.2, 0.8],  # Lavender
                ]
            )

        label = self.class_l.view(-1).long()
        valid = label != -1

        colours = cmap.to(self.device)[label[valid]]
        return o3d_cloud(self.xyz[valid], colours=colours)

    def as_o3d_trunk_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter((self.branch_ids == trunk_id).view(-1)).as_o3d_cld()

    def as_o3d_branch_cld(self) -> o3d.geometry.PointCloud:
        trunk_id = self.branch_ids[0]
        return self.filter((self.branch_ids != trunk_id).view(-1)).as_o3d_cld()

    def as_o3d_class_loss_mask_cld(self) -> o3d.geometry.PointCloud:
        cmap = torch.tensor([[1, 0, 0], [0, 1, 0]])
        colours = cmap.to(self.device)[self.class_loss_mask.long().view(-1)]
        return o3d_cloud(self.xyz, colours=colours)

    def as_o3d_vector_loss_mask_cld(self) -> o3d.geometry.PointCloud:
        cmap = torch.tensor([[1, 0, 0], [0, 1, 0]])
        colours = cmap.to(self.device)[self.vector_loss_mask.long().view(-1)]
        return o3d_cloud(self.xyz, colours=colours)

    def as_o3d_medial_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.medial_pts)

    def as_o3d_medial_vectors(self) -> o3d.geometry.LineSet:
        return o3d_lines_between_clouds(self.as_o3d_cld(), self.as_o3d_medial_cld())

    @property
    def viewer_items(self) -> list[ViewerItem]:
        items = super().viewer_items
        item = partial(ViewerItem, is_visible=False, group=f"{super().group_name}")
        if self.medial_vector is not None:
            items += [item("Medial Vectors", self.as_o3d_medial_vectors())]
            items += [item("Medial Cloud", self.as_o3d_medial_cld())]
        if self.branch_ids is not None:
            items += [item("Trunk", self.as_o3d_trunk_cld())]
            items += [item("Branches", self.as_o3d_branch_cld())]
        if self.class_l is not None:
            items += [item("Segmented", self.as_o3d_segmented_cld())]
        if self.class_loss_mask is not None:
            items += [item("Class Loss Mask", self.as_o3d_class_loss_mask_cld())]
        if self.vector_loss_mask is not None:
            items += [item("Vector Loss Mask", self.as_o3d_vector_loss_mask_cld())]
        return items

    def view(self):
        o3d_viewer(self.viewer_items)