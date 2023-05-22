import cugraph
import cupy
import frnn
import numpy as np
import pandas as pd
import torch
from cudf import DataFrame
from tqdm import tqdm


def knn(src, dest, K=50, r=1.0, grid=None):
    src_lengths = src.new_tensor([src.shape[0]], dtype=torch.long)
    dest_lengths = src.new_tensor([dest.shape[0]], dtype=torch.long)
    dists, idxs, grid, _ = frnn.frnn_grid_points(
        src.unsqueeze(0),
        dest.unsqueeze(0),
        src_lengths,
        dest_lengths,
        K,
        r,
        return_nn=False,
        return_sorted=True,
    )

    return idxs.squeeze(0), dists.sqrt().squeeze(0), grid


def nn(src, dest, r=1.0, grid=None):
    idx, dist, grid = knn(src, dest, K=1, r=r, grid=grid)
    idx, dist = idx.squeeze(1), dist.squeeze(1)

    return idx, dist, grid


def nn_graph(points: torch.Tensor, radii, K=40):
    idxs, dists, _ = knn(points, points, K=K, r=radii.max().item())
    idxs[
        (dists > radii.unsqueeze(1)) & (dists > 0)
    ] = -1  # We don't want edges forming to itself...
    return make_edges(dists, idxs)


# NN graph except links that don't exist get weighted super heavily (edge length outside radius)...
def nn_graph_distance_weighted(points: torch.Tensor, radii, K=40):
    idxs, dists, _ = knn(points, points, K=K, r=200)

    dists[dists > radii.unsqueeze(1)] = (
        radii.max().item() + dists[dists > radii.unsqueeze(1)] ** 2
    )

    return make_edges(dists, idxs)


def medial_nn_graph(points: torch.Tensor, radii, medial_dist, K=40):
    # edges weighted based on distance to medial axis
    idxs, dists, _ = knn(points, points, K=K, r=radii.max().item() / 4.0)
    dists_ = dists + medial_dist[idxs]  # Add medial distance to distance graph...
    idxs[dists > radii.unsqueeze(1)] = -1

    return make_edges(dists_, idxs)


def make_edges(dists, idxs):
    n = dists.shape[0]
    K = dists.shape[1]

    parent = torch.arange(n, device=dists.device).unsqueeze(1).expand(n, K)
    edges = torch.stack([parent, idxs], dim=2)

    valid = idxs.view(-1) > 0
    return edges.view(-1, 2)[valid], dists.view(-1)[valid]


def nn_flat(points, K=50, r=1.0, device=torch.device("cuda")):
    idxs, dists = nn(points, K, r, device)
    return idxs.reshape(-1), dists.reshape(-1)


def pcd_nn(points, radii, K=20):
    idxs, dists = nn_flat(np.asarray(points), K=K, r=float(radii.max() / 4.0))

    parent_vertex = np.repeat(np.arange(points.shape[0]), K)
    parent_vertex = parent_vertex.reshape(-1)

    edges = np.vstack((parent_vertex, idxs.reshape(-1))).T

    valid = idxs != -1
    return edges[valid], dists[valid]


def cuda_graph(edges, edge_weights, renumber=False):
    edges = cupy.asarray(edges)
    edge_weights = cupy.asarray(edge_weights)

    d = DataFrame()
    d["source"] = edges[:, 0]
    d["destination"] = edges[:, 1]
    d["weights"] = edge_weights
    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(d, edge_attr="weights", renumber=renumber)

    return g


def decompose_cuda_graph(cuda_graph, device):
    pdf = cugraph.to_pandas_edgelist(cuda_graph)

    edges = torch.stack((torch.tensor(pdf["src"]), torch.tensor(pdf["dst"])), dim=1)
    edge_weights = torch.tensor(pdf["weights"])

    # last_edge = torch.tensor(
    #    (edges.shape[0]-1, edges.shape[0]-1), device=device).unsqueeze(0)
    # edges = torch.cat((edges, last_edge), dim=0)

    # edge_weights = torch.tensor(pdf["weights"])
    # edge_weights = torch.cat((edge_weights, torch.zeros(1, device=device)), dim=0).float()

    return edges.long().to(device), edge_weights.to(device)


def connected_components(edges, edge_weights, minimum_vertices=0, max_components=10):
    g = cuda_graph(edges, edge_weights)

    connected_components = cugraph.connected_components(g)

    num_labels = connected_components["labels"].to_pandas().value_counts()
    valid_labels = num_labels[num_labels > minimum_vertices].index

    graphs = []

    for label in tqdm(
        valid_labels[:max_components], desc="Getting Connected Componenets", leave=False
    ):
        graphs.append(
            cugraph.subgraph(
                g, connected_components.query(f"labels == {label}")["vertex"]
            )
        )

    return graphs


def pcd_to_tetra(points):
    """(Delaunay Triangulation)"""
    delaunay_tetra = pytetgen.Delaunay(points)
    simplices = delaunay_tetra.simplices
    edges = np.vstack(
        (
            np.column_stack((simplices[:, 0], simplices[:, 1])),
            np.column_stack((simplices[:, 1], simplices[:, 2])),
            np.column_stack((simplices[:, 2], simplices[:, 3])),
            np.column_stack((simplices[:, 3], simplices[:, 0])),
        )
    )
    edge_weights = np.sqrt(
        np.sum(
            (delaunay_tetra.points[edges[:, 0]] - delaunay_tetra.points[edges[:, 1]])
            ** 2,
            axis=1,
        )
    )
    return edges, edge_weights
