from pathlib import Path
from typing import List

import torch

from smart_tree.data_types.cloud import Cloud


def model_output_to_labelled_clds(
    sparse_input,
    model_output,
    cmap,
    filenames,
) -> List[Cloud]:
    return to_labelled_clds(
        sparse_input.indices[:, 0],
        sparse_input.features[:, :3],
        sparse_input.features[:, 3:6],
        model_output,
        cmap,
        filenames,
    )


def split_outputs(features, mask):
    radii = torch.exp(features["radius"][mask])
    direction = features["medial_direction"][mask]
    class_l = torch.argmax(features["class_l"], dim=1)[mask]

    return radii, direction, class_l


def to_labelled_clds(
    cloud_ids,
    coords,
    rgb,
    model_output,
    cmap,
    filenames,
) -> List[Cloud]:
    num_clouds = cloud_ids.max() + 1
    clouds = []

    # assert rgb.shape[1] > 0

    for i in range(num_clouds):
        mask = cloud_ids == i
        xyz = coords[mask]
        rgb = torch.rand(xyz.shape)  # rgb[mask]

        radii, direction, class_l = split_outputs(model_output, mask)

        labelled_cloud = Cloud(
            xyz=xyz,
            rgb=rgb,
            medial_vector=radii * direction,
            class_l=class_l,
            filename=Path(filenames[i]),
        )

        clouds.append(labelled_cloud.to_device(torch.device("cpu")))

    return clouds
