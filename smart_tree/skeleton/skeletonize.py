import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from cugraph import sssp
from tqdm import tqdm

from ..data_types.tree import TreeSkeleton, DisjointTreeSkeleton
from ..util.mesh.geometries import (
    o3d_cloud,
    o3d_line_set,
    o3d_merge_clouds,
    o3d_merge_linesets,
    o3d_path,
)
from .filter import outlier_removal
from .graph import (
    connected_components,
    decompose_cuda_graph,
    nn_graph,
    nn_graph_distance_weighted,
    pcd_to_tetra,
)
from .path import sample_tree
from .shortest_path import edge_graph, graph_shortest_paths, pred_graph, shortest_paths
from ..data_types.tree import TreeSkeleton
from ..util.visualizer.view import o3d_viewer
from ..data_types.cloud import LabelledCloud


class Skeletonizer:
    def __init__(
        self,
        K: int,  # how many neighbours to connect to
        min_connection_length: float,  # connect all points to other points withint (min_edge, r)
        minimum_graph_vertices: int,  # remove all sub-graphs that have vertices less than this
        voxel_downsample: float,  # how much to voxel downsample medial points
        edge_non_linear: float,
        max_number_components: int,
        device=torch.device("cuda:0"),
    ):
        print("Initalizing Skeletonizer")

        self.K = K
        self.min_connection_length = min_connection_length
        self.minimum_graph_vertices = minimum_graph_vertices
        self.voxel_downsample = voxel_downsample
        self.max_number_components = max_number_components
        self.device = device

    def forward(self, cloud: LabelledCloud):
        cloud.to_device(self.device)

        if self.voxel_downsample != False:
            cloud = cloud.medial_voxel_down_sample(self.voxel_downsample)

        mask = outlier_removal(cloud.medial_pts, cloud.radii.unsqueeze(1))
        cloud = cloud.filter(mask)

        edges, edge_weights = nn_graph(
            cloud.medial_pts,
            cloud.radii.clamp(min=self.min_connection_length),
            K=self.K,
        )
        subgraphs = connected_components(
            edges,
            edge_weights,
            minimum_vertices=self.minimum_graph_vertices,
            max_components=self.max_number_components,
        )

        return DisjointTreeSkeleton(
            skeletons=[
                self.process_subgraph(
                    subgraph,
                    cloud.medial_pts,
                    cloud.xyz,
                    cloud.radii.unsqueeze(1),
                    self.min_connection_length,
                    self.device,
                    skeleton_id,
                )
                for skeleton_id, subgraph in enumerate(
                    tqdm(subgraphs, desc="Processing Skeleton Fragment", leave=False)
                )
            ]
        )

    def process_subgraph(
        self,
        subgraph,
        medial_points,
        points,
        radii,
        min_edge,
        device=torch.device("cuda:0"),
        skeleton_id=0,
    ) -> TreeSkeleton:
        edges, edge_weights = decompose_cuda_graph(subgraph, self.device)

        root_idx = edges[torch.argmin(medial_points[edges[:, 0]][:, 1])][0].item()

        verts, preds, distance = shortest_paths(
            root_idx,
            edges,
            edge_weights,
            renumber=False,
        )

        predecessor_graph = pred_graph(verts, preds, medial_points)

        distances = torch.as_tensor(
            sssp(predecessor_graph, source=root_idx)["distance"]
        ).to(self.device)

        branches = sample_tree(medial_points, radii, preds, distances, points)

        return TreeSkeleton(skeleton_id, branches)

    @staticmethod
    def from_cfg(cfg):
        return Skeletonizer(
            K=cfg.K,
            min_connection_length=cfg.min_connection_length,
            minimum_graph_vertices=cfg.minimum_graph_vertices,
            voxel_downsample=cfg.voxel_downsample,
            edge_non_linear=cfg.edge_non_linear,
            max_number_components=cfg.max_number_components,
        )


if __name__ == "__main__":
    main()
