from dataclasses import asdict, dataclass
from typing import List

import cugraph
import cupy
import open3d as o3d
import torch
from cudf import DataFrame
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_line_set
from ..o3d_abstractions.visualizer import ViewerItem, o3d_viewer

patch_typeguard()


@typechecked
@dataclass
class Graph:
    vertices: TensorType["N", 3]
    edges: TensorType["N", 2]
    edge_weights: TensorType["N", 1]

    def __len__(self) -> int:
        return self.edges.shape[0]

    def __str__(self) -> str:
        return (
            f"{'*' * 80}"
            f"Graph :\n"
            f"Num Edges: {self.edges.shape[0]}\n"
            f"Num Vertices: {self.vertices.shape[0]}\n"
            f"{'*' * 80}"
        )

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

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.to(device)

        return Graph(**args)

    def as_o3d_lineset(self, colour=(1, 0, 0)) -> o3d.geometry.LineSet:
        return o3d_line_set(self.vertices, self.edges, colour=colour)

    def viewer_items(self) -> list[ViewerItem]:
        return [ViewerItem(f"Graph Lineset", self.as_o3d_lineset())]

    def view(self):
        o3d_viewer(self.viewer_items())

    # def connected_cugraph_components(
    #     self,
    #     minimum_vertices: int = 10,
    # ) -> List[cugraph.Graph]:
    #     g = cuda_graph(self.edges, self.edge_weights)
    #     df = cugraph.connected_components(g)

    #     components = [
    #         cugraph.subgraph(g, subgraph_vertices)
    #         for _, subgraph_vertices in tqdm(
    #             df.query(f"vertex >= {minimum_vertices}")["vertex"].unique().to_pandas(),
    #             desc="Finding Connected Components",
    #             leave=False,
    #         )
    #     ]

    #     return sorted(components, key=lambda graph: len(graph.nodes()), reverse=True)


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
