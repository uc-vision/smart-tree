import os
from collections import defaultdict

import numpy as np
import open3d as o3d
import torch

from smart_tree.data_types.cloud import LabelledCloud, merge_labelled_cloud
from smart_tree.util.file import save_cloud

if __name__ == "__main__":
    directory = "/home/harry/Desktop/latest_row/labelled/"
    ply_files = [f for f in os.listdir(directory) if f.endswith(".ply")]

    point_clouds = defaultdict(list)
    for filename in ply_files:
        point_cloud_id, class_name = (
            filename.split("_")[0],
            filename.rsplit("_", 1)[1].split(".")[0],
        )

        point_clouds[point_cloud_id].append(
            (class_name, os.path.join(directory, filename))
        )

    class_to_label = {
        "Unlabelled": -1,
        "Ground": 5,
        "Post": 4,
        "Trunk": 1,
    }

    class_to_color = {
        "Unlabelled": (1.0, 1.0, 0.0),
        "Ground": (0.0, 0.0, 1.0),
        "Post": (0.0, 1.0, 1.0),
        "Trunk": (1.0, 0.0, 1.0),
    }

    for cloud_id, fpath_cloud_list in point_clouds.items():
        labelled_clouds = []
        for class_name, fpath in fpath_cloud_list:
            o3d_cloud = o3d.io.read_point_cloud(fpath)

            xyz = torch.tensor(np.array(o3d_cloud.points))
            rgb = torch.tensor(np.array(o3d_cloud.colors))

            num_points = xyz.shape[0]

            labelled_clouds.append(
                LabelledCloud(
                    xyz=xyz,
                    rgb=rgb,
                    # medial_vector=torch.full((num_points, 3), -1),
                    # branch_direction=torch.full((num_points, 3), -1),
                    class_l=torch.full(
                        (num_points, 1),
                        class_to_label[class_name],
                    ).long(),
                )
            )

        labelled_cloud = merge_labelled_cloud(labelled_clouds)

        save_cloud(labelled_cloud, f"{directory}/{cloud_id}.npz")
        # o3d_viewer(
        #     labelled_cloud.viewer_items,
        # )
