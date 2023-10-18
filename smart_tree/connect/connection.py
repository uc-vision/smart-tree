from typing import List
import torch
import networkx as nx

from smart_tree.data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer, ViewerItem
from smart_tree.o3d_abstractions.geometries import o3d_cloud, o3d_merge_clouds
from smart_tree.util.maths import torch_dot

from smart_tree.data_types.graph import Graph

from torch.nn import functional as F

from tqdm import tqdm


def weighted_nx_graph(weighted_edges):
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    return G


def bellman_ford_distance(graph, root_node):
    return nx.single_source_bellman_ford_path_length(graph, root_node)


def path_distance(nx_graph, node_distances, source_idx=0):
    node_degrees = nx_graph.degree()
    terminal_nodes = [n for n, d in node_degrees if d == 1]
    junction_nodes = [n for n, d in nx_graph.degree() if d not in [1, 2]]

    dfs_edges = list(nx.dfs_edges(nx_graph, source=source_idx))

    path_distances = {}

    dist = 0
    for edge in dfs_edges:
        node = edge[1]

        dist += node_distances[node]

        if node in junction_nodes:
            if node in path_distances:
                dist = path_distances[node]
            else:
                path_distances[node] = dist

        path_distances[node] = dist

        if node in terminal_nodes:
            dist = 0

    return path_distances


class SkeletonConnector:
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.device = device

    def forward(self, skeleton: DisjointTreeSkeleton) -> TreeSkeleton:
        self.graphs: List[Graph] = [s.to_graph() for s in skeleton.skeletons]

        root_node_idxs = []
        for g in tqdm(self.graphs, desc="computing edge weights"):
            weighted_edges, node_distances = self.compute_edge_weights(g)
            nx_graph = weighted_nx_graph(weighted_edges)

            path_dists = path_distance(nx_graph, node_distances, source_idx=0)
            closest_node_id, max_distance = min(path_dists.items(), key=lambda x: x[1])

            closest_node_edges = nx_graph.edges(closest_node_id, data=True)

            for e1, e2, ew in closest_node_edges:
                if e1 == closest_node_edges:
                    if ew["weight"] < 0:
                        closest_node_edges = e2

                if e2 == closest_node_edges:
                    if ew["weight"] > 0:
                        closest_node_edges = e1

            root_node_idxs.append(closest_node_id)

        viewer_items = skeleton.viewer_items
        cld = ViewerItem(
            "cld",
            o3d_merge_clouds(
                [
                    o3d_cloud(s.vertices[i].cpu().numpy().reshape(-1, 3))
                    for s, i in zip(self.graphs, root_node_idxs)
                ]
            ),
        )

        viewer_items.append(cld)

        o3d_viewer(viewer_items)
        return True

    def compute_edge_weights(self, graph, source_node=0):
        nx_graph = graph.as_networkx_graph()
        preds = nx.predecessor(nx_graph, source_node)

        weighted_edges = []
        node_weights = {}

        for node, predecessor in preds.items():
            if predecessor == []:
                continue
            node = int(node)
            predecessor = int(predecessor[0])

            direction = graph.vertices[node] - graph.vertices[predecessor]
            norm_dir = F.normalize(direction.unsqueeze(0))
            branch_dir = F.normalize(graph.branch_direction[predecessor].unsqueeze(0))
            ew = float(torch_dot(norm_dir, branch_dir))

            weighted_edges.append((predecessor, node, ew))

            node_weights[node] = ew

        return weighted_edges, node_weights

        v = [
            graph.vertices[int(k)] - graph.vertices[int(v[0])]
            for k, v in preds.items()[1:]
        ]
        print(v)
        traversal_direcions = F.normalize(torch.conv)

        # branch_directions = F.normalize(graph.branch_direction[int(k)])

        # print(traversal_direcionts)

        quit()

        path_direction = F.normalize(
            self.graph.vertices[verts[1:]] - self.graph.vertices[preds[1:]]
        )
        branch_direction = F.normalize(graph.branch_direction[verts[1:]])

        new_edge_weights = (
            torch_dot(path_direction, -branch_direction).clamp(min=1e-16) ** 2
        )

    def find_graph_root_node_idx(graph: Graph):
        pass


if __name__ == "__main__":
    skeleton = DisjointTreeSkeleton.from_pickle(
        "/local/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
    )
    print("loaded")
    connector = SkeletonConnector()

    connector.forward(skeleton)
    quit()

    # print("yeyeye")
    # viewer_items = skeleton.viewer_items
    # o3d_viewer(viewer_items)
    # g = skeleton.as_nextworkx_graph()

    graphs = [s.to_graph() for s in skeleton.skeletons]

    start_idxs = [g.vertices.shape[0] // 2 for g in graphs]
    nextworkx_graphs = [g.as_networkx_graph() for g in graphs]
    distances = [
        nx.single_source_bellman_ford_path_length(g, idx)
        for g, idx in tqdm(zip(nextworkx_graphs, start_idxs), desc="Finding root nodes")
    ]

    furthest_nodes = [int(max(d, key=d.get)) for d in distances]

    for s in graphs:
        print(s.vertices)

    cld = ViewerItem(
        "cld",
        o3d_merge_clouds(
            [
                o3d_cloud(s.vertices[i].cpu().numpy().reshape(-1, 3))
                for s, i in zip(graphs, furthest_nodes)
            ]
        ),
    )

    print(furthest_nodes)

    viewer_items.append(cld)

    o3d_viewer(viewer_items)

    quit(0)

    main_graph = graphs.pop(0)
