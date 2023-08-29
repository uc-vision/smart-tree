import torch

from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.o3d_abstractions.geometries import (
    o3d_cloud,
    o3d_elipsoid,
    o3d_merge_meshes,
)
from smart_tree.util.file import load_cloud

from smart_tree.skeleton.graph import knn, make_edges, nn_graph

from smart_tree.data_types.graph import Graph

from pathlib import Path

from tqdm import tqdm

import numpy as np

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
def pairwise_points_inside_ellipsoids(
    pts: TensorType["N", 3],
    ellipsoid_centres: TensorType["N", 3],
    ellipsoid_semi_axis: TensorType["N", 3],
    eps=1e-12,
) -> TensorType["N"]:
    #
    query = ((pts - ellipsoid_centres) / (ellipsoid_semi_axis + eps)).square().sum(1)
    return query <= 1.0


@typechecked
def ellipsoid_graph(
    points: TensorType["N", 3],
    radii: TensorType["N"],
    branch_directions: TensorType["N", 3],
    K=16,
    stretch_factor=2,
):
    # This is not optimal as it requires a sufficient K

    ellipsoid_semi_axis = (
        radii.unsqueeze(1) * branch_directions * stretch_factor
    ) + radii.unsqueeze(1)
    ellipsoid_semi_axis = torch.abs(ellipsoid_semi_axis)

    idxs, dists, _ = knn(points, points, K=K, r=radii.max().item() * stretch_factor)

    inside = pairwise_points_inside_ellipsoids(
        points.repeat_interleave(K, dim=0),
        points[idxs].reshape(-1, 3),
        ellipsoid_semi_axis[idxs].reshape(-1, 3),
    )

    idxs[~inside.reshape(-1, K)] = -1

    edges, edge_weights = make_edges(dists, idxs)
    return Graph(points, edges, edge_weights)


def debug():
    pts = torch.tensor([[0, 0, 0], [1, 0, 0]])
    ellipsoid_centres = np.array([[0, 0, 0], [1, 0, 0]])
    ellipsoid_semi_axis = np.array([[0.5, 0.5, 0.5], [1.0, 0.1, 0.1]])

    o3d_viewer(
        [
            o3d_cloud(pts),
            o3d_elipsoid(ellipsoid_semi_axis[1], ellipsoid_centres[1]),
            o3d_elipsoid(ellipsoid_semi_axis[0], ellipsoid_centres[0]),
        ]
    )


if __name__ == "__main__":
    cld = load_cloud(Path("/local/UC-10/npz_4mm/apple/apple_1.npz"))
    print("cloud loaded")

    g = ellipsoid_graph(
        cld.medial_pts.cuda(),
        cld.radius.cuda(),
        cld.branch_direction.cuda(),
        K=16,
        stretch_factor=3,
    )

    g2 = nn_graph(
        cld.medial_pts.cuda(),
        cld.radius.cuda(),
        K=16,
    )

    g3 = nn_graph(
        cld.medial_pts.cuda(),
        cld.radius.cuda() * 2,
        K=16,
    )

    o3d_viewer(
        [
            g.to_o3d_lineset(),
            g2.to_o3d_lineset().paint_uniform_color([0, 1, 0]),
            g3.to_o3d_lineset().paint_uniform_color([0, 0, 1]),
            cld.to_o3d_cld(),
            cld.to_o3d_medial_vectors(),
        ]
    )

    # o3d_elipses = []

    # for c, d, r in tqdm(zip(centres, directions, radii)):
    #     elipse_params = r.repeat(3) + (np.abs(d) * r * 2)  # r.repeat(3)  # + (d * r)

    #     o3d_elipses.append(o3d_elipsoid(elipse_params.numpy(), c.numpy()))

    # o3d_viewer([o3d_merge_meshes(o3d_elipses), cld.to_o3d_cld()])

    # print(centres)

    # quit()

    # cld.medial_pts

    # radius = np.array([0.1])

    # direction = np.array([1, 0.0, 0.0])

    # elipse_params = radius.repeat(3) + (direction * radius)

    # elipse = o3d_elipsoid(elipse_params)

    # o3d_viewer([elipse])
