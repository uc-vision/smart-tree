from typing import List

import cugraph
import cupy
import torch
from cugraph import sssp
from tqdm import tqdm

from ..data_types.cloud import LabelledCloud
from ..data_types.graph import Graph
from ..data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from .graph import decompose_cuda_graph, nn_graph
from .path import sample_tree
from .shortest_path import pred_graph, shortest_paths


class Skeletonizer:
    def __init__(
        self,
        K: int,
        min_connection_length: float,
        minimum_graph_vertices: int,
        outlier_remove_nb_points: int = 8,
        prune_skeletons=False,
        min_skeleton_radius=None,
        min_skeleton_length=None,
        repair_skeletons=False,
        smooth_skeletons=False,
        smooth_kernel_size=11,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.K = K
        self.min_connection_length = min_connection_length
        self.minimum_graph_vertices = minimum_graph_vertices
        self.outlier_remove_nb_points = outlier_remove_nb_points

        self.prune_skeletons = prune_skeletons
        self.min_skeleton_radius = min_skeleton_radius
        self.min_skeleton_length = min_skeleton_length
        self.repair_skeletons = repair_skeletons
        self.smooth_skeletons = smooth_skeletons
        self.smooth_kernel_size = smooth_kernel_size

        self.device = device

    def forward(self, cloud: LabelledCloud) -> DisjointTreeSkeleton:
        cloud.to_device(self.device)

        # mask = outlier_removal(
        #     cloud.medial_pts,
        #     cloud.radius,
        #     nb_points=self.outlier_remove_nb_points,
        # )
        # cloud = cloud.filter(mask)

        graph: Graph = nn_graph(
            cloud.medial_pts,
            cloud.radius.clamp(min=self.min_connection_length),
            K=self.K,
        )

        subgraphs: List[cugraph.Graph] = graph.connected_cugraph_components(
            minimum_vertices=self.minimum_graph_vertices
        )

        skeletons = []

        pbar = tqdm(subgraphs, desc="Skeletonizing", leave=False)
        for subgraph_id, subgraph in enumerate(pbar):
            skeleton = self.process_subgraph(cloud, subgraph, subgraph_id)
            if len(skeleton.branches) != 0:
                skeletons.append(skeleton)

        skeleton = DisjointTreeSkeleton(skeletons)

        if self.prune_skeletons:
            skeleton.prune(
                min_length=self.min_skeleton_length,
                min_radius=self.min_skeleton_radius,
            )

        if self.repair_skeletons:
            skeleton.repair()

        if self.smooth_skeletons:
            skeleton.smooth(self.smooth_kernel_size)

        return skeleton

    def process_subgraph(
        self,
        cloud: LabelledCloud,
        subgraph,
        skeleton_id=0,
    ) -> TreeSkeleton:
        """Extract skeleton for connected component"""

        subgraph_vertice_idx = torch.tensor(
            cupy.unique(subgraph.edges().values),
            device=self.device,
        )

        subgraph_cloud = cloud.filter(subgraph_vertice_idx)

        edges, edge_weights = decompose_cuda_graph(
            subgraph,
            renumber_edges=True,
            device=self.device,
        )

        verts, preds, distances = shortest_paths(
            subgraph_cloud.root_idx,
            edges,
            edge_weights,
            renumber=False,
        )

        predecessor_graph = pred_graph(verts, preds, subgraph_cloud.medial_pts)

        distances = torch.as_tensor(
            sssp(predecessor_graph, source=subgraph_cloud.root_idx)["distance"],
            device=self.device,
        )

        branches = sample_tree(
            subgraph_cloud,
            preds,
            distances,
        )

        return TreeSkeleton(skeleton_id, branches, colour=torch.rand(3))


#     @staticmethod
#     def from_cfg(cfg):
#         return Skeletonizer(
#             K=cfg.K,
#             min_connection_length=cfg.min_connection_length,
#             minimum_graph_vertices=cfg.minimum_graph_vertices,
#             edge_non_linear=cfg.edge_non_linear,
#         )


# if __name__ == "__main__":
#     main()
