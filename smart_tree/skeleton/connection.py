import torch
from typing import List
import cugraph
import cudf
import open3d as o3d
from tqdm import tqdm
from smart_tree.data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.util.queries import (
    pts_to_pts_squared_distances,
    skeleton_to_skeleton_distance,
)

from torch.nn import functional as F
from smart_tree.util.maths import torch_dot

from smart_tree.skeleton.shortest_path import shortest_paths
from smart_tree.data_types.graph import Graph


def find_closest_skeleton(skeleton: TreeSkeleton, query_skeletons: List[TreeSkeleton]):
    min_dist = torch.inf

    for query_skeleton in tqdm(query_skeletons):
        dist = torch.min(skeleton_to_skeleton_distance(skeleton, query_skeleton))

        if dist < min_dist:
            min_dist = dist
            closest_skeleton = query_skeleton

    return closest_skeleton


def find_skeleton_base(skeleton: TreeSkeleton):
    graph = skeleton.to_graph()

    # graph.view()

    # graph_cuda = graph.as_cugraph()
    # initial_root_idx = 0

    # paths = cugraph.sssp(graph_cuda, initial_root_idx)

    start_idx = 150

    verts, preds, distance = shortest_paths(
        start_idx,
        graph.edges,
        graph.edge_weights,
        renumber=False,
    )

    print(f"xyz {graph.vertices.shape}")
    print(f"edges {graph.edges.shape}")
    print(f"ew {graph.edge_weights.shape}")

    print(f"verts {verts.shape}")
    print(f"preds {preds.shape}")
    print(f"distance {distance.shape}")

    path_direction = F.normalize(graph.vertices[verts[1:]] - graph.vertices[preds[1:]])
    branch_direction = F.normalize(graph.branch_direction[verts[1:]])

    print(torch_dot(path_direction, -branch_direction))

    new_edge_weights = (
        torch_dot(path_direction, -branch_direction).clamp(min=1e-16) ** 2
    )

    graph = Graph(graph.vertices, graph.edges, new_edge_weights)
    graph2 = Graph(
        graph.vertices.cpu(),
        graph.edges[(new_edge_weights.cpu() > 0.5).reshape(-1)],
        new_edge_weights[(new_edge_weights.cpu() > 0.5).reshape(-1)],
    )
    start_sphere = (
        o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        .translate(graph.vertices[start_idx].cpu())
        .paint_uniform_color((1, 0, 0))
    )

    # graph.view()

    # graph2.view()

    og_graph = graph.as_o3d_lineset()
    og_graph2 = graph2.as_o3d_lineset()

    o3d_viewer([og_graph, og_graph2.paint_uniform_color((1, 0, 0)), start_sphere])

    # ear
    # new_edge_weights = torch.ones(graph.edges.shape[1], 1).cuda()

    print(new_edge_weights.shape)
    print(graph.edges.shape)

    verts, preds, distance = shortest_paths(
        start_idx,
        graph.edges,
        new_edge_weights,
        renumber=False,
    )

    print(distance)

    base_idx = distance.argmax()

    base_sphere = [
        o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        .translate(graph.vertices[base_idx].cpu())
        .paint_uniform_color((0, 1, 0))
    ]

    print(f"base idx is {base_idx}")
    o3d_viewer(graph.viewer_items + base_sphere + [start_sphere])

    # graph_v1 = graph.xyz[self.edges[0]]
    # graph_v2 = graph.xyz[self.edges[1]]


if __name__ == "__main__":
    print("loading")
    disjoint_skeleton = DisjointTreeSkeleton.from_pickle(
        "/local/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
    )
    print("loaded")

    print("sorting")
    skeletons_sorted = sorted(
        disjoint_skeleton.skeletons,
        key=lambda x: x.length,
        reverse=True,
    )[0:2]
    print("sorted")

    main_skeleton = skeletons_sorted[0]

    find_skeleton_base(main_skeleton)

    quit()

    closest_skeleton = find_closest_skeleton(main_skeleton, skeletons_sorted[1:])

    find_skeleton_base(closest_skeleton)

    quit()

    for branch_id, branch in closest_skeleton.branches.items():
        print(branch.branch_direction)

    print(main_skeleton)

    o3d_viewer(main_skeleton.viewer_items + closest_skeleton.viewer_items)

    print("skeleton loaded")

    # # disjoint_skeleton.prune(min_radius=0.01, min_length=0.08).smooth(kernel_size=11)
    # # Sort skeletons by total length

    # skeletons_sorted[0].view()
    # skeletons_sorted[1].view()

    # quit()

    # skel = connect(skeletons_sorted[0], 0, 0, skeletons_sorted[1], 0, 0)

    # skel.view()

    # quit()

    # final_skeleton = TreeSkeleton(0, skeletons_sorted[0].branches)

    # for skeleton in skeletons_sorted[1:]:
    #     branch = skeleton.branches[skeleton.key_branch_with_biggest_radius]

    #     # get the point that has the biggest radius ....
    #     # get the closest point on the skeleton to that point ...
    #     # connect the two points ...

    #     print(skeleton.length)

    #     o3d_viewer(
    #         [final_skeleton.to_o3d_tube(), branch.to_o3d_tube(), skeleton.to_o3d_tube()]
    #     )
