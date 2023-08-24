import cmapy
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from .geometries import o3d_cloud, o3d_clouds, o3d_line_set
from .camera import o3d_headless_render


def o3d_viewer(items, names=[], line_width=1):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = line_width

    geometries = []
    if len(names) == 0:
        names = np.arange(0, len(items))

    for name, item in zip(names, items):
        if type(item) == o3d.geometry.LineSet:
            geometries.append(
                {
                    "name": f"{name}",
                    "geometry": item,
                    "material": line_mat,
                    "is_visible": False,
                }
            )
        else:
            geometries.append(
                {
                    "name": f"{name}",
                    "geometry": item,
                    "material": mat,
                    "is_visible": False,
                }
            )

    o3d.visualization.draw(geometries, line_width=line_width)
