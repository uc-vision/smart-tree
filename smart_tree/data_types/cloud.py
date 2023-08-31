from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from smart_tree.o3d_abstractions.visualizer import ViewerItem

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from ..util.misc import voxel_downsample
from ..util.queries import skeleton_to_points
from .tree import TreeSkeleton

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
            f"Coloured: {hasattr(self, 'rgb')}\n"
            f"Filename: {self.filename}\n"
            f"{'*' * 80}"
        )

    def scale(self, factor: float) -> Cloud:
        scaled_xyz = self.xyz * factor
        return Cloud(xyz=scaled_xyz, rgb=self.rgb, filename=self.filename)

    def translate(self, translation_vector: TensorType[3, float]) -> Cloud:
        translated_xyz = self.xyz + translation_vector
        return Cloud(xyz=translated_xyz, rgb=self.rgb, filename=self.filename)

    def rotate(self, rotation_matrix: TensorType[3, 3, float]) -> Cloud:
        rotated_xyz = torch.matmul(self.xyz, rotation_matrix.T)
        return Cloud(xyz=rotated_xyz, rgb=self.rgb, filename=self.filename)

    def filter(self, mask: TensorType["N", bool]) -> Cloud:
        filtered_rgb = self.rgb[mask] if self.rgb is not None else None
        return Cloud(xyz=self.xyz[mask], rgb=filtered_rgb, filename=self.filename)

    def to_device(self, device: torch.device):
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
        return torch.max(self.xyz, 0)[0]

    @property
    def min_xyz(self) -> TensorType[3]:
        return torch.min(self.xyz, 0)[0]

    @property
    def bounding_box(self) -> tuple[TensorType[3], TensorType[3]]:
        # defined by centre coordinate, x/2, y/2, z/2
        dimensions = (self.max_xyz - self.min_xyz) / 2
        centre = self.min_xyz + dimensions
        return centre, dimensions

    def as_o3d_cld(self) -> o3d.geometry.PointCloud:
        return o3d_cloud(self.xyz, self.rgb)

    def view_items(self) -> list[ViewerItem]:
        return [ViewerItem("Cloud", self.as_o3d_cld(), is_visible=True)]

    def view(self) -> None:
        o3d_viewer(self.view_items())


@typechecked
@dataclass
class LabelledCloud(Cloud):
    """
    Medial Vector: Vector towards medial axis
    Branch Direction: Unit vector of branch direction of closest branch
    Branch ID: ID of closest branch
    Class: Class of point (I.E. 0 = trunk, 1 = branch, 2 = leaf)
    """

    medial_vector: Optional[TensorType["N", 3]] = None
    branch_direction: Optional[TensorType["N", 3]] = None
    branch_ids: Optional[TensorType["N", 1]] = None
    class_l: Optional[TensorType["N", 1]] = None

    def scale(self, factor: float) -> LabelledCloud:
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
                args["branch_direction"], rot_matrix.T
            )
        return LabelledCloud(**args)

    def filter(self, mask: TensorType["N", torch.bool]) -> LabelledCloud:
        args = asdict(self)
        for k, v in args.items():
            if v is not None and isinstance(v, torch.Tensor):
                args[k] = v[mask]
        return LabelledCloud(**args)

    def filter_by_class(self, classes: TensorType["N"]) -> LabelledCloud:
        mask = torch.isin(self.class_l, classes)
        return self.filter(mask.view(-1))

    def pin_memory(self):
        args = asdict(self)
        for k, v in args.items():
            if v is not None and isinstance(v, torch.Tensor):
                args[k] = v.pin_memory()
        return LabelledCloud(**args)

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            if v is not None and isinstance(v, torch.Tensor):
                args[k] = v.to(device)

        return LabelledCloud(**args)

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return torch.unique(self.class_l).item() + 1

    @property
    def radius(self) -> TensorType["N", 1]:
        return self.medial_vector.pow(2).sum(1).sqrt()

    @property
    def medial_direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector)

    def as_o3d_segmented_cld(
        self, cmap: TensorType["N", 3] = None
    ) -> o3d.geometry.PointCloud:
        if cmap is None:
            cmap = torch.rand(self.number_classes, 3)
        colours = cmap[self.class_l.view(-1).int()]
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

    def view_items(self) -> list[ViewerItem]:
        items = super().view_items()
        if self.medial_vector is not None:
            items += [ViewerItem("Medial Vectors", self.as_o3d_medial_vectors())]
        if self.branch_direction is not None:
            items += [ViewerItem("Branch Directions", self.as_o3d_branch_directions())]
        if self.branch_ids is not None:
            items += [ViewerItem("Trunk", self.as_o3d_trunk_cld())]
            items += [ViewerItem("Branches", self.as_o3d_branch_cld())]
        if self.class_l is not None:
            items += [ViewerItem("Segmented", self.as_o3d_segmented_cld())]
        return items

    def view(self):
        o3d_viewer(self.view_items())


class CloudLoader:
    def load(self, npz_file: Union[str, Path]) -> Union[Cloud, LabelledCloud]:
        # Load the NPZ file
        data = np.load(npz_file)

        # Check if any of the optional parameters exist to determine the type
        optional_params = [
            f.name for f in fields(LabelledCloud) if f.default is not None
        ]

        for param in optional_params:
            if param in data:
                return self._load_as_labelled_cloud(data)

        return self._load_as_cloud(data)

    def _load_as_cloud(self, data) -> Cloud:
        # Filter the fields that exist in the Cloud class
        cloud_fields = {f.name: data[f.name] for f in fields(Cloud) if f.name in data}

        return Cloud(**cloud_fields)

    def _load_as_labelled_cloud(self, data) -> LabelledCloud:
        # Filter the fields that exist in the LabelledCloud class
        labelled_cloud_fields = {
            f.name: data[f.name] for f in fields(LabelledCloud) if f.name in data
        }

        return LabelledCloud(**labelled_cloud_fields)


#     def to_device(self, device):
#         xyz = self.xyz.to(device)
#         rgb = self.rgb.to(device) if self.rgb is not None else None
#         medial_vector = (
#             self.medial_vector.to(device) if self.medial_vector is not None else None
#         )
#         branch_direction = (
#             self.branch_direction.to(device)
#             if self.branch_direction is not None
#             else None
#         )
#         class_l = self#     def filter_by_class(self, classes):
#         classes = torch.tensor(classes, device=self.class_l.device)
#         mask = torch.isin(
#             self.class_l,
#             classes,
#         )
#         return self.filter(mask.view(-1))

#     def filter_by_skeleton(skeleton: TreeSkeleton, cloud: Cloud, threshold=1.1):
#         distances, radii, vectors_ = skeleton_to_points(cloud, skeleton, chunk_size=512)
#         mask = distances < radii * threshold
#         return filter(cloud, mask)

#     def pin_memory(self):
#         properties = {
#             "xyz": self.xyz.pin_memory(),
#             "rgb": self.rgb.pin_memory() if self.rgb is not None else None,
#             "medial_vector": self.medial_vector.pin_memory()
#             if self.medial_vector is not None
#             else None,
#             "branch_direction": self.branch_direction.pin_memory()
#             if self.branch_direction is not None
#             else None,
#             "class_l": self.class_l.pin_memory() if self.class_l is not None else None,
#             "branch_ids": self.branch_ids.pin_memory()
#             if self.branch_ids is not None
#             else None,
#         }

#         return Cloud(**properties)

#     def filter(cloud: Cloud, mask: TensorType["N"]) -> Cloud:
#         mask = mask.to(cloud.xyz.device)

#         filtered_properties = {
#             "xyz": cloud.xyz[mask],
#             "rgb": cloud.rgb[mask] if cloud.rgb is not None else None,
#             "medial_vector": cloud.medial_vector[mask]
#             if cloud.medial_vector is not None
#             else None,
#             "branch_direction": cloud.branch_direction[mask]
#             if cloud.branch_direction is not None
#             else None,
#             "class_l": cloud.class_l[mask] if cloud.class_l is not None else None,
#             "branch_ids": cloud.branch_ids[mask]
#             if cloud.branch_ids is not None
#             else None,
#             "filename": cloud.filename if cloud.filename is not None else None,
#         }

#         return Cloud(**filtered_properties)

#     def cpu(self):
#         return self.to_device(torch.device("cpu"))
#             rgb=rgb,
#             medial_vector=medial_vector,
#             branch_direction=branch_direction,
#             class_l=class_l,
#             branch_ids=branch_ids,
#             filename=filename,
#         )

#     def view(self, cmap=[]):
#         # if cmap == []:
#         cmap = np.random.rand(self.number_classes, 3)

#         cpu_cld = self.cpu()
#         geoms = []

#         geoms.append(cpu_cld.to_o3d_cld())
#         if cpu_cld.class_l != None:
#             geoms.append(cpu_cld.to_o3d_seg_cld(cmap))

#         if cpu_cld.medial_vector != None:
#             projected = o3d_cloud(cpu_cld.xyz + cpu_cld.medial_vector, colour=(1, 0, 0))
#             geoms.append(projected)
#             geoms.append(o3d_lines_between_clouds(cpu_cld.to_o3d_cld(), projected))

#         o3d_viewer(geoms)

#     def voxel_down_sample(self, voxel_size):
#         idx = voxel_downsample(self.xyz, voxel_size)
#         return self.filter(idx)

#     def scale(self, factor):
#         new_medial_vector = None
#         if self.medial_vector != None:
#             new_medial_vector = self.medial_vector * factor

#         return Cloud(
#             xyz=self.xyz * factor,
#             rgb=self.rgb,
#             medial_vector=new_medial_vector,
#             branch_direction=self.branch_direction,
#             branch_ids=self.branch_ids,
#             class_l=self.class_l,
#         )

#     def translate(self, xyz):
#         return Cloud(
#             xyz=self.xyz + xyz,
#             rgb=self.rgb,
#             medial_vector=self.medial_vector,
#             branch_direction=self.branch_direction,
#             branch_ids=self.branch_ids)

#     @staticmethod
#     def from_o3d_cld(cld) -> Cloud:
#         return Cloud.from_numpy(xyz=np.asarray(cld.points), rgb=np.asarray(cld.colors))
#             class_l=self.class_l,
#         )

#     def rotate(self, rot_mat):
#         new_xyz = torch.matmul(self.xyz, rot_mat.to(self.xyz.device))

#         new_medial_vector = None
#         if self.medial_vector != None:
#             new_medial_vector = torch.matmul(
#                 self.medial_vector, rot_mat.to(self.medial_vector.device)
#             )

#         new_branch_dir = None
#         if self.branch_direction != None:
#             new_branch_dir = torch.matmul(
#                 self.branch_direction, rot_mat.to(self.branch_direction.device)
#             )

#         return Cloud(
#             xyz=new_xyz,
#             rgb=self.rgb,
#             medial_vector=new_medial_vector,
#             branch_direction=new_branch_dir,
#             branch_ids=self.branch_ids,
#             class_l=self.class_l,
#         )

#     @property
#     def is_labelled(self) -> bool:
#         if self.medial_vector is None:
#             return False
#         return True

#     @property
#     def root_idx(self) -> int:
#         return torch.argmin(self.xyz[:, 1]).item()

#     @property
#     def number_classes(self) -> int:
#         if not hasattr(self, "class_l"):
#             return 1
#         return torch.max(self.class_l).item() + 1

#     @property
#     def max_xyz(self) -> torch.Tensor:
#         return torch.max(self.xyz, 0)[0]

#     @property
#     def min_xyz(self) -> torch.Tensor:
#         return torch.min(self.xyz, 0)[0]

#     @property
#     def device(self) -> torch.device:
#         return self.xyz.device

#     @property
#     def bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
#         # defined by centre coordinate, x/2, y/2, z/2
#         dimensions = (self.max_xyz - self.min_xyz) / 2
#         centre = self.min_xyz + dimensions
#         return centre, dimensions

#     @property
#     def medial_pts(self) -> torch.Tensor:
#         return self.xyz + self.medial_vector

#     @staticmethod
#     def from_numpy(**kwargs) -> "Cloud":
#         torch_kwargs = {}

#         for key, value in kwargs.items():
#             if key in [
#                 "xyz",
#                 "rgb",
#                 "medial_vector",
#                 "branch_direction",
#                 "branch_ids",
#                 "class_l",
#             ]:
#                 torch_kwargs[key] = torch.tensor(value).float()

#             """ SUPPORT LEGACY NPZ -> Remove in Future..."""
#             if key in ["vector"]:
#                 torch_kwargs["medial_vector"] = torch.tensor(value).float()

#         return Cloud(**torch_kwargs)

#     @property
#     def radius(self) -> TensorType["N", 1] | torch.Tensor:
#         return self.medial_vector.pow(2).sum(1).sqrt().unsqueeze(1)

#     @property
#     def medial_direction(self) -> torch.Tensor:
#         return F.normalize(self.medial_vector)

#     def with_xyz(self, new_xyz):
#         return Cloud(
#             new_xyz,
#             self.rgb,
#             self.medial_vector,
#             self.branch_direction,
#             self.branch_ids,
#             self.class_l,
#             self.filename,
#         )
