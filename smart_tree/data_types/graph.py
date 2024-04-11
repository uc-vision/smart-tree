from dataclasses import dataclass
from typing import List, Optional

import cugraph
import cupy
import networkx as nx
import numpy as np
import open3d as o3d
import torch
from cudf import DataFrame
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_line_set
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from .base import Base

patch_typeguard()


@typechecked
@dataclass
class Graph(Base):
    vertices: TensorType["M", 3]
    edges: TensorType["N", 2]
    edge_weights: TensorType["N", 1]

    radius: Optional[TensorType["M", 1, float]] = None
    branch_direction: Optional[TensorType["M", 3, float]] = None
    root_idx: TensorType[1, int] = None

    def __len__(self) -> int:
        return self.vertices.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Graph :\n"
            f"Num Edges: {self.edges.shape[0]}\n"
            f"Num Vertices: {self.vertices.shape[0]}\n"
            f"{'*' * 80}"
        )

    def add_edge(self, edge: TensorType[1, 2, int], weight: TensorType[1, 1, float]):
        self.edges = torch.cat([self.edges, edge], dim=0)
        self.edge_weights = torch.cat([self.edge_weights, weight], dim=0)

    def connected_cugraph_components(
        self,
        minimum_vertices: int = 10,
    ) -> List[cugraph.Graph]:
        g = cuda_graph(self.edges, self.edge_weights)
        df = cugraph.connected_components(g)

        components = []
        for label in tqdm(
            df["labels"].unique().to_pandas(),
            desc="Finding Connected Components",
            leave=False,
        ):
            subgraph_vertices = df[df["labels"] == label]["vertex"]

            if subgraph_vertices.count() < minimum_vertices:
                continue

            components.append(cugraph.subgraph(g, subgraph_vertices))

        return sorted(components, key=lambda graph: len(graph.nodes()), reverse=True)

    def as_o3d_lineset(
        self,
        colour=(1, 0, 0),
        colours=None,
        heat_map=True,
    ) -> o3d.geometry.LineSet:
        if heat_map:
            colours = torch.rand(self.vertices.shape)

        return o3d_line_set(self.vertices, self.edges, colour=colour, colours=colours)

    def as_cugraph(self):
        return cuda_graph(self.edges, self.edge_weights)

    def as_networkx_graph(self):
        G = nx.Graph()

        weighted_edges = np.hstack((self.edges, self.edge_weights))
        G.add_weighted_edges_from(weighted_edges)
        return G

    @property
    def edge_vertices(self) -> TensorType["N", 2, 3, float]:
        return self.vertices[self.edges]

    @property
    def viewer_items(self) -> list[ViewerItem]:
        return [ViewerItem(f"Graph Lineset", self.as_o3d_lineset())]

    def view(self):
        o3d_viewer(self.viewer_items)


def cuda_graph(edges, edge_weights, renumber=False):
    edges = cupy.asarray(edges)
    edge_weights = cupy.asarray(edge_weights.reshape(-1))

    d = DataFrame()
    d["source"] = edges[:, 0]
    d["destination"] = edges[:, 1]
    d["weights"] = edge_weights
    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(d, edge_attr="weights", renumber=renumber)

    return g


def join_graphs(graphs: List[Graph], offset_edges=False) -> Graph:
    combined_vertices = torch.cat([graph.vertices for graph in graphs], dim=0)

    offset = 0
    combined_edges = []
    for graph in graphs:
        combined_edges.append(graph.edges + offset)
        if offset_edges:
            offset += graph.vertices.shape[0]

    combined_edges = torch.cat(combined_edges, dim=0)
    combined_edge_weights = torch.cat([graph.edge_weights for graph in graphs], dim=0)
    combined_branch_direction = (
        torch.cat([graph.branch_direction for graph in graphs], dim=0)
        if any(graph.branch_direction is not None for graph in graphs)
        else None
    )

    return Graph(
        vertices=combined_vertices,
        edges=combined_edges,
        edge_weights=combined_edge_weights,
        branch_direction=combined_branch_direction,
    )
