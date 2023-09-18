from __future__ import annotations

from typing import List

import numpy as np
import torch
from tqdm import tqdm
import math

from smart_tree.data_types.tube import CollatedTube, Tube, collate_tubes
from smart_tree.data_types.line import (
    collates_line_segments,
    LineSegment,
    CollatedLineSegment,
)


"""
For the following :
  N : number of pts
  M : number of tubes
"""


# GPU


def pts_to_pts_squared_distances(pts1, pts2):
    return torch.sum((pts1.unsqueeze(1) - pts2.unsqueeze(0)) ** 2, dim=2)


def points_to_collated_tube_projections(pts: torch.tensor, collated_tube: CollatedTube):
    ab = collated_tube.b - collated_tube.a  # M x 3

    ap = pts.unsqueeze(1) - collated_tube.a.unsqueeze(0)  # N x M x 3

    t = (torch.einsum("nmd,md->nm", ap, ab) / torch.einsum("md,md->m", ab, ab)).clip(
        0.0,
        1.0,
    )  # N x M

    proj = collated_tube.a.unsqueeze(0) + torch.einsum("nm,md->nmd", t, ab)  # N x M x 3
    return proj, t


def projection_to_distance_matrix(projections, pts):  # N x M x 3
    return (projections - pts.unsqueeze(1)).square().sum(2).sqrt()


def pts_to_nearest_tube(
    pts: torch.tensor,
    tubes: List[Tube],
    device=torch.device("cuda"),
):
    """Vectors from pt to the nearest tube"""

    collated_tube = collate_tubes(tubes)
    collated_tube_gpu = collated_tube.to_device(device)

    pts = pts.float().to(device)

    projections, t = points_to_collated_tube_projections(
        pts, collated_tube_gpu
    )  # N x M x 3
    r = (1 - t) * collated_tube_gpu.r1.squeeze(1) + t * collated_tube_gpu.r2.squeeze(1)
    distances = projection_to_distance_matrix(projections, pts)  # N x M
    distances = torch.abs(distances - r)
    idx = torch.argmin(distances, 1)  # N

    assert idx.shape[0] == pts.shape[0]

    return (
        projections[torch.arange(pts.shape[0]), idx] - pts,
        idx,
        r[torch.arange(pts.shape[0]), idx],
    )


def projection_to_distance_matrix_keops(projections, pts):  # N x M x 3
    return np.sqrt(np.sum(np.square(projections - pts[:, np.newaxis, :]), 2))  # N x M


def skeleton_to_points(pcd, skeleton, chunk_size=4096, device="gpu"):
    all_distances = []
    all_radii = []
    all_vectors = []
    all_idx = []

    tubes = skeleton.to_tubes()
    pts_chunks = torch.chunk(
        pcd.xyz, chunks=math.ceil(pcd.xyz.shape[0] / chunk_size), dim=1
    )

    for pts in tqdm(pts_chunks, desc="Labelling Chunks", leave=False):
        vectors, idxs, radiuses = pts_to_nearest_tube(pts, tubes)

        all_distances.append(torch.sqrt(torch.einsum("ij,ij->i", vectors, vectors)))
        all_radii.append(radiuses)
        all_vectors.append(vectors)
        all_idx.append(idxs)

    all_distances = torch.cat(all_distances)
    all_radii = torch.cat(all_radii)
    all_vectors = torch.cat(all_vectors)
    all_idx = torch.cat(all_idx)

    print(all_vectors.shape)

    return all_distances, all_radii, all_vectors, all_idx


def distance_between_line_segments_and_tubes(
    line_segments: List[LineSegment], tubes: List[Tube], device=torch.device("cuda")
):
    tubes: CollatedTube = collate_tubes(tubes)
    lines: CollatedLineSegment = collates_line_segments(line_segments)

    tubes.to_device(device)
    lines.to_device(device)

    line_directions = lines.b - lines.a
    line_lengths = torch.norm(lines.b - lines.a)
    line_directions /= line_lengths  # Nx3

    # N x 3 - M X 3 -> N x M X 3
    a_diff = lines.a[:, None, :] - tubes.a[None, :, :]

    # Projection of tube onto line
    ## N x M -> Dot product ->
    t = torch.clip(
        torch.sum(a_diff * line_directions.unsqueeze(1), axis=2),
        0,
        1,
    )

    ## N X M x 3 -> closest point on line
    closest_pt_on_lines = lines.a.unsqueeze(1) + t.unsqueeze(2) * line_lengths.reshape(
        1, 1, -1
    )

    distance_vectors = closest_pt_on_lines - tubes.a.unsqueeze(0)

    ## N x M
    distance_along_axis = torch.clip(
        torch.sum(distance_vectors * tubes.b.unsqueeze(0) - tubes.a.unsqueeze(0), 2)
        / line_lengths,
        0,
        1,
    )

    # N x 3 + N x M * N x 3
    closest_pts_on_tube_axes = tubes.a.unsqueeze(0) + distance_along_axis.unsqueeze(
        2
    ) * (tubes.b - tubes.a).unsqueeze(0)

    distances_to_tube_axes = torch.norm(
        closest_pt_on_lines - closest_pts_on_tube_axes,
        dim=2,
    )

    r = (1 - t) * tubes.r1.squeeze(1) + t * tubes.r2.squeeze(1)

    return (distances_to_tube_axes - r).cpu()


def skeleton_to_skeleton_distance(
    skel1: TreeSkeleton,
    skel2: TreeSkeleton,
    device=torch.device("cuda"),
):
    skel1_line = skel1.to_line_segments()
    skel2_tubes = skel2.to_tubes()

    return distance_between_line_segments_and_tubes(skel1_line, skel2_tubes)
