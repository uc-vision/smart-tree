import math
from typing import List, Union

import sys
import cmapy
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def to_torch(numpy_arrays: List[np.array], device=torch.device("cpu")):
    return [torch.from_numpy(np_arr).float().to(device) for np_arr in numpy_arrays]


def to_numpy(_torch: Union[List[torch.tensor], torch.tensor]):
    if type(_torch) is list:
        return [torch_arr.cpu().detach().numpy() for torch_arr in _torch]
    else:
        return _torch.cpu().detach().numpy()


def concate_dict_of_tensors(tensors: dict, device=torch.device("cpu")):
    for key, values in tensors.items():
        tensors[key] = torch.concatenate(values).to(device)
    return tensors


def unique_n_colours(num_colours, cmap="hsv"):
    return (
        np.asarray(
            [cmapy.color(cmap, i) for i in range(0, 255, math.ceil(255 / num_colours))]
        ).reshape(-1, 3)
        / 255
    )


def unique_n_random_colours(num_colours):
    return np.asarray([np.random.rand(3) for i in range(num_colours)]).reshape(-1, 3)


def points_to_edges(points):
    points = points.reshape(-1, 3)
    parents = torch.arange(points.shape[0] - 1)
    children = torch.arange(1, points.shape[0])

    return torch.column_stack((parents, children))


def voxel_downsample(xyz, voxel_size):
    xyz_quantized = (
        xyz // voxel_size
    )  # torch.div(xyz + (voxel_size / 2), voxel_size, rounding_mode="floor")

    unique, idx, counts = torch.unique(
        xyz_quantized,
        dim=0,
        sorted=True,
        return_counts=True,
        return_inverse=True,
    )

    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum[1:]]

    return first_indicies
