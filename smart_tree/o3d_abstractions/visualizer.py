import cmapy
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Union, List, Sequence, Tuple, Optional

from .geometries import o3d_cloud, o3d_clouds, o3d_line_set
from .camera import o3d_headless_render


@dataclass
class ViewerItem:
    name: str
    geometry: o3d.geometry.Geometry
    is_visible: bool = True


def o3d_viewer(
    items: Union[Sequence[ViewerItem], List[o3d.geometry.Geometry]], line_width=1
):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = line_width

    if isinstance(items, list):
        items = [ViewerItem(f"{i}", item) for i, item in enumerate(items)]

    def material(item):
        return line_mat if isinstance(item.geometry, o3d.geometry.LineSet) else mat

    geometries = [dict(**asdict(item), material=material(item)) for item in items]
    o3d.visualization.draw(geometries, line_width=line_width)
