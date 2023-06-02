from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.figure_factory as ff
import torch
import torch.nn.functional as F
from pykeops.numpy import LazyTensor as LazyTensor_np
from pykeops.torch import LazyTensor
from tqdm import tqdm

from smart_tree.data_types.tube import CollatedTube, Tube, collate_tubes
from smart_tree.util.misc import to_numpy

""" 
For the following :
  N : number of pts
  M : number of tubes
"""


def points_to_collated_tube_projections(
    pts: np.array, collated_tube: CollatedTube, eps=1e-12
):  # N x 3, M x 2
    ab = collated_tube.b - collated_tube.a  # M x 3

    ap = pts[:, np.newaxis] - collated_tube.a[np.newaxis, ...]  # N x M x 3

    t = np.clip(
        np.einsum("nmd,md->nm", ap, ab) / (np.einsum("md,md->m", ab, ab) + eps),
        0.0,
        1.0,
    )  # N x M
    proj = collated_tube.a[np.newaxis, ...] + np.einsum(
        "nm,md->nmd", t, ab
    )  # N x M x 3
    return proj, t


def projection_to_distance_matrix(projections, pts):  # N x M x 3
    return np.sqrt(np.sum(np.square(projections - pts[:, np.newaxis, :]), 2))  # N x M


def pts_to_nearest_tube(pts: np.array, tubes: List[Tube]):
    """Vectors from pt to the nearest tube"""

    collated_tube = collate_tubes(tubes)
    projections, t = points_to_collated_tube_projections(
        pts, collated_tube
    )  # N x M x 3

    r = (1 - t) * collated_tube.r1 + t * collated_tube.r2

    distances = projection_to_distance_matrix(projections, pts)  # N x M

    distances = distances - r
    idx = np.argmin(distances, 1)  # N

    # assert idx.shape[0] == pts.shape[0]

    return (
        projections[np.arange(pts.shape[0]), idx] - pts,
        idx,
        r[np.arange(pts.shape[0]), idx],
    )  # vector, idx , radius


def pairwise_pts_to_nearest_tube(pts: np.array, tubes: List[Tube]):
    collated_tube = collate_tubes(tubes)

    ab = collated_tube.b - collated_tube.a  # M x 3
    ap = pts - collated_tube.a  # N x 3

    t = ((ap * ab).sum(1) ** 0.5) / ((ab**2).sum(1) ** 0.5)
    proj = collated_tube.a + ab * t.reshape(-1, 1)

    r = (1 - t) * collated_tube.r1 + t * collated_tube.r2

    distances = np.sqrt(np.sum(np.square(proj - pts), 1))

    distances = distances - r

    return distances, r  # vector, idx , radius


# GPU
def points_to_collated_tube_projections_gpu(
    pts: np.array, collated_tube: CollatedTube, device=torch.device("cuda")
):
    ab = collated_tube.b - collated_tube.a  # M x 3

    ap = pts.unsqueeze(1) - collated_tube.a.unsqueeze(0)  # N x M x 3

    t = (torch.einsum("nmd,md->nm", ap, ab) / torch.einsum("md,md->m", ab, ab)).clip(
        0.0, 1.0
    )  # N x M
    proj = collated_tube.a.unsqueeze(0) + torch.einsum("nm,md->nmd", t, ab)  # N x M x 3
    return proj, t


def projection_to_distance_matrix_gpu(projections, pts):  # N x M x 3
    return (projections - pts.unsqueeze(1)).square().sum(2).sqrt()


def pts_to_nearest_tube_gpu(
    pts: torch.tensor, tubes: List[Tube], device=torch.device("cuda")
):
    """Vectors from pt to the nearest tube"""

    collated_tube_gpu = collate_tubes(tubes)
    collated_tube_gpu.to_gpu()

    pts = pts.float().to(device)

    projections, t = points_to_collated_tube_projections_gpu(
        pts, collated_tube_gpu, device=torch.device("cuda")
    )  # N x M x 3
    r = (1 - t) * collated_tube_gpu.r1 + t * collated_tube_gpu.r2

    distances = projection_to_distance_matrix_gpu(projections, pts)  # N x M

    distances = torch.abs(distances - r)
    idx = torch.argmin(distances, 1)  # N

    assert idx.shape[0] == pts.shape[0]

    return (
        projections[torch.arange(pts.shape[0]), idx] - pts,
        idx,
        r[torch.arange(pts.shape[0]), idx],
    )


"""
  KeOps
"""


def distance_matrix_keops(pts1, pts2):
    x_i = LazyTensor(pts1.reshape(-1, 1, 3))
    y_j = LazyTensor(pts2.view(1, -1, 3))

    return (x_i - y_j).square().sum(dim=2).sqrt()


def nn_keops(pts1, pts2):
    D_ij = distance_matrix_keops(pts1, pts2)

    return D_ij.min(1), D_ij.argmin(1).flatten()  # distance, idx


def points_to_tube_distance_keops(
    pts: np.array, tubes: List[Tube], eps=1e-12
):  # N x 3, M x 2
    collated_tube = collate_tubes(tubes)

    # M number of lines
    ab = (collated_tube.b - collated_tube.a)[np.newaxis, ...]  # 1 x M x 3
    ap = (pts[:, np.newaxis] - collated_tube.a[np.newaxis, ...])[
        :, np.newaxis, ...
    ]  # 1 x N x M x 3

    a_lzy = LazyTensor_np(collated_tube.a[np.newaxis, ...])
    ab_lzy = LazyTensor_np(ab)
    ap_lzy = LazyTensor_np(ap)

    t_lzy = ((ab_lzy * ap_lzy).sum(3) / (ab_lzy * ab_lzy).sum(2)).clamp(0.0, 1.0)
    proj_lzy = a_lzy + (t_lzy * ab_lzy)

    r1_lzy = LazyTensor_np(collated_tube.r1[..., np.newaxis])
    r2_lzy = LazyTensor_np(collated_tube.r2[..., np.newaxis])

    r_lzy = (1 - t_lzy) * r1_lzy + t_lzy * r2_lzy

    pts_lzy = LazyTensor_np(pts.reshape(-1, 1, 1, 3))

    dist_lzy = (proj_lzy - pts_lzy).square().sum(3).sqrt()  # .square().sum(2).sqrt())

    dist_lzy = dist_lzy - r_lzy

    idxs = dist_lzy.argmin(2)  # idx of the closest line segment for each point

    tubes = [tubes[idx[0]] for idx in idxs[:, :, 0]]

    distances, r = pairwise_pts_to_nearest_tube(pts, tubes)

    return distances, idxs, r


def projection_to_distance_matrix_keops(projections, pts):  # N x M x 3
    return np.sqrt(np.sum(np.square(projections - pts[:, np.newaxis, :]), 2))  # N x M


def pts_to_nearest_tube_keops(pts: np.array, tubes: List[Tube]):
    """Vectors from pt to the nearest tube"""

    distances, idx, r = points_to_tube_distance_keops(pts, tubes)  # N x M x 3

    return distances.reshape(-1), idx, r


def skeleton_to_points(pcd, skeleton, chunk_size=4096, device="gpu"):
    distances = []
    radii = []
    vectors_ = []

    tubes = skeleton.to_tubes()
    pts_chunks = np.array_split(pcd.xyz, np.ceil(pcd.xyz.shape[0] / chunk_size))

    for pts in tqdm(pts_chunks, desc="Labelling Chunks", leave=False):
        if device == "gpu":
            vectors, idxs, radiuses = pts_to_nearest_tube_gpu(
                pts, tubes
            )  # vector to nearest skeleton...
        else:
            vectors, idxs, radiuses = pts_to_nearest_tube(
                pts, tubes
            )  # vector to nearest skeleton...

        distances.append(
            np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
        )  # could do on gpu but meh
        radii.append([radius for radius in radiuses])
        vectors_.append([v for v in vectors])

    distances = np.concatenate(distances)
    radii = np.concatenate(radii)
    vectors_ = np.concatenate(vectors_)

    return distances, radii, vectors_
