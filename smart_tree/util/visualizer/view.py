import cmapy
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from smart_tree.util.mesh.geometries import o3d_cloud, o3d_clouds, o3d_line_set
from smart_tree.util.visualizer.camera import o3d_headless_render


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


def view_segmentation_results(
    xyz, model_prediction, class_colours, camera_position, camera_up
):
    predicted_class = torch.argmax(model_prediction, dim=1).numpy()
    class_colours = np.asarray(class_colours)

    colours = class_colours[predicted_class]

    cloud = o3d_cloud(xyz, colours=colours)

    geoms = [cloud]

    return np.asarray(o3d_headless_render(geoms, camera_position, camera_up))


def view_point_projections(xyz, model_prediction, camera_position, camera_up):
    input_cloud = o3d_cloud(xyz, colour=(1, 0, 0))

    rad_pred = np.exp(model_prediction[:, 0].cpu().numpy()).reshape(-1, 1)
    direction_pred = model_prediction[:, 1:].cpu().numpy()
    medial_estimate = xyz + rad_pred * direction_pred

    shifted_cloud = o3d_cloud(medial_estimate, colour=(0, 0, 1))

    geoms = [input_cloud, shifted_cloud]

    return np.asarray(o3d_headless_render(geoms, camera_position, camera_up))
