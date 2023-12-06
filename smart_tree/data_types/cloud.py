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
from ..util.misc import voxel_filter
from .base import Base

patch_typeguard()


@typechecked
@dataclass
class Cloud(Base):
    xyz: TensorType["N", 3, torch.float32]
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

    def voxel_downsample(self, voxel_size: float | TensorType[1]) -> Cloud:
        return self.filter(voxel_filter(self.xyz, voxel_size).reshape(-1))

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
    def centre(self) -> TensorType[3]:
        return torch.mean(self.xyz, dim=0)

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
    A class representing a point cloud with additional labeled data.
    """

    medial_vector: Optional[TensorType["N", 3]] = None
    """ Vector towards medial axis. """

    branch_direction: Optional[TensorType["N", 3]] = None
    """ Unit vector of branch direction of closest branch. """

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
        if args["medial_vector"] is not None:
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
        rot_mat = rot_matrix.T.to(self.device).to(self.xyz.dtype)

        args["xyz"] = torch.matmul(args["xyz"], rot_mat).to(self.xyz.dtype)
        if args["medial_vector"] is not None:
            args["medial_vector"] = torch.matmul(args["medial_vector"], rot_mat)
        if args["branch_direction"] is not None:
            args["branch_direction"] = torch.matmul(
                args["branch_direction"],
                rot_mat,
            )
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
        return F.normalize(self.medial_vector)

    @property
    def medial_pts(self) -> TensorType["N", 3]:
        return self.xyz + self.medial_vector

    def as_o3d_segmented_cld(
        self,
        cmap: TensorType["N", 3] = None,
    ) -> o3d.geometry.PointCloud:
        if cmap is None:
            cmap = torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # Trunk
                    [0.0, 1.0, 0.0],  # Spur /\ Cane /\ Shoot
                    [0.0, 0.0, 1.0],  # Node
                    [1.0, 1.0, 0.0],  # Wire
                    [0.0, 1.0, 1.0],  # Post
                    [1.0, 0.5, 1.0],
                ]
            )

        colours = cmap.to(self.device)[self.class_l.view(-1).long()]
        return o3d_cloud(self.xyz, colours=colours)

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
        if self.class_loss_mask is not None:
            items += [item("Class Loss Mask", self.as_o3d_class_loss_mask_cld())]
        if self.vector_loss_mask is not None:
            items += [item("Vector Loss Mask", self.as_o3d_vector_loss_mask_cld())]
        return items

    def view(self):
        o3d_viewer(self.viewer_items)


def concatenate_tensors(
    tensors: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    filtered_tensors = [
        tensor
        for tensor in tensors
        if tensor is not None and isinstance(tensor, torch.Tensor)
    ]
    return torch.cat(filtered_tensors, dim=0) if filtered_tensors else None


def pad_tensor_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.shape[0] == target_length:
        return tensor

    remainder = target_length - tensor.shape[0]
    padding = torch.full(
        (remainder, *tensor.shape[1:]), torch.nan, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=0)


def merge_labelled_cloud(clouds: List[LabelledCloud]) -> LabelledCloud:
    if not clouds:
        raise ValueError("The input list of clouds is empty")

    field_names = [field.name for field in fields(LabelledCloud)]

    # Concatenate tensors for each field
    concatenated = {
        field: concatenate_tensors(
            [
                getattr(cloud, field)
                for cloud in clouds
                if getattr(cloud, field) is not None
            ]
        )
        for field in field_names
    }

    # Find the target length from the 'xyz' attribute
    target_length = (
        concatenated["xyz"].shape[0] if concatenated.get("xyz") is not None else None
    )

    # Pad other tensors to match the target length
    for field in field_names:
        if field == "xyz" or concatenated[field] is None:
            continue

        if concatenated[field].shape[0] != target_length:
            concatenated[field] = pad_tensor_to_length(
                concatenated[field], target_length
            )

    return LabelledCloud(**concatenated)
