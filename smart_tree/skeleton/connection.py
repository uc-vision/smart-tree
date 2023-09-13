from typing import List
from dataclasses import dataclass


import torch
from torch.nn import functional as F
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.skeleton.shortest_path import shortest_paths
from smart_tree.util.maths import torch_dot, magnitudes
from smart_tree.util.queries import skeleton_to_points, skeleton_to_skeleton_distance
from smart_tree.data_types.graph import join_graphs, Graph
from smart_tree.data_types.branch import BranchSkeleton


@dataclass
class SkeletonBase:
    idx: int
    vertice: torch.Tensor


@dataclass
class SkeletonBases:
    bases: List[SkeletonBase]

    def __len__(self):
        return len(self.bases)

    def as_cloud(self):
        return Cloud(torch.stack([b.vertice for b in self.bases]))

    def pop(self, idx):
        return self.bases.pop(idx)


def find_closest_skeleton(
    skeleton: TreeSkeleton,
    query_skeletons: List[TreeSkeleton],
) -> int:
    min_dist = torch.inf
    closest_skeleton_idx = -1

    for idx, query_skeleton in tqdm(enumerate(query_skeletons)):
        dist = torch.min(skeleton_to_skeleton_distance(skeleton, query_skeleton))

        if dist < min_dist:
            min_dist = dist
            closest_skeleton_idx = idx

    return closest_skeleton_idx, query_skeletons[closest_skeleton_idx]


def find_graph_root_idx(graph: Graph):
    start_idx = graph.vertices.shape[0] // 2

    verts, preds, distance = shortest_paths(
        start_idx,
        graph.edges,
        graph.edge_weights,
        renumber=False,
    )

    path_direction = F.normalize(graph.vertices[verts[1:]] - graph.vertices[preds[1:]])
    branch_direction = F.normalize(graph.branch_direction[verts[1:]])

    new_edge_weights = (
        torch_dot(path_direction, -branch_direction).clamp(min=1e-16) ** 2
    )

    verts, preds, distance = shortest_paths(
        start_idx,
        graph.edges,
        new_edge_weights,
        renumber=False,
    )
    return distance.argmax()

    # return SkeletonBase(idx, graph.vertices[idx])


def find_graph_root_nodes(graphs: List[Graph]):
    return [find_graph_root_idx(g) for g in tqdm(graphs)]

    # return SkeletonBases([find_skeleton_base(s) for s in tqdm(skeleton)])


def sort_skeletons_by_length(skeletons: List[TreeSkeleton]):
    return sorted(
        skeletons,
        key=lambda x: x.length,
        reverse=True,
    )


def find_closest_skeleton_base(skeleton: TreeSkeleton, cld: Cloud):
    distances, radii, vectors, idxs = skeleton_to_points(cld, skeleton)
    cld_idx = distances.argmin()
    return cld_idx, vectors[cld_idx], idxs[cld_idx]


def connect_graphs(
    g1: Graph,
    g2: Graph,
    g1_vert_idx: int,
    g2_vert_idx: int,
    edge_weight: float,
):
    new_edge = torch.tensor([g1_vert_idx, g2_vert_idx + g1.vertices.shape[0]]).reshape(
        -1, 2
    )

    g1.add_edge(new_edge, torch.tensor(edge_weight).reshape(-1, 1))

    return join_graphs([g1, g2], offset_edges=True)


def find_nearest_graph_point(graph1: Graph, graph2: Graph, graph_2_root_idx):
    graph2_root_vert = graph2.vertices[graph_2_root_idx]
    graph2_root_branch_direction = graph2.branch_direction[graph_2_root_idx]

    graph1_verts = graph1.vertices

    potential_connection_dirs = F.normalize(graph1_verts - graph2_root_vert)

    direction_weighting = torch_dot(
        potential_connection_dirs,
        graph2_root_branch_direction,
    )

    distance = magnitudes(graph1_verts - graph2_root_vert)

    edge_weighting = distance  # * -direction_weighting

    return edge_weighting.argmin()


if __name__ == "__main__":
    viewer_items = []

    disjoint_skeleton = DisjointTreeSkeleton.from_pickle(
        "/local/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
    )

    graphs = [s.to_graph() for s in disjoint_skeleton.skeletons[0:2]]
    root_idxs = torch.tensor(find_graph_root_nodes(graphs))
    roots_verts = torch.cat(
        [g.vertices[[idx.item()]] for idx, g in zip(root_idxs, graphs)], dim=0
    )

    g1_connection_idx = find_nearest_graph_point(graphs[0], graphs[1], root_idxs[1])

    graph = connect_graphs(graphs[0], graphs[1], g1_connection_idx, root_idxs[1], 1.0)

    graph.view()

    # graphs[0].add_edge(root_idxs + max_idx, connect_idx)

    # print(roots_verts)

    quit()

    main_graph = graphs.pop(0)

    # graphs[0].view()

    quit()

    skeletons_sorted = sort_skeletons_by_length(disjoint_skeleton.skeletons)

    skeleton = skeletons_sorted.pop(0)

    graph = skeleton.to_graph()

    skeleton_bases = find_skeleton_bases(skeletons_sorted)

    while len(skeleton_bases) > 0:
        base_idx, vec2skel, skeleton_idx = find_closest_skeleton_base(
            skeleton,
            skeleton_bases.as_cloud(),
        )
        viewer_items += skeleton_bases.as_cloud().viewer_items

        closest_base = skeleton_bases.pop(base_idx)
        closest_skeleton = skeletons_sorted.pop(base_idx)

        BranchSkeleton()

        closest_skeleton.add_branch()

        pt1 = pt1 + vec2skel

        connection_branch = skeleton.branch_from_tube_idx(skeleton_idx.cpu())
        connection_branch_2 = closest_skeleton.branch_from_tube_idx(base_idx.cpu())
        # base_branch = closest_skeleton

        viewer_items += connection_branch.viewer_items
        viewer_items += connection_branch_2.viewer_items

        viewer_items += skeleton.viewer_items
        viewer_items += closest_skeleton.viewer_items

        o3d_viewer(viewer_items)
        # graph = join_graphs([graph, closest_skeleton.to_graph()], True)

        # graph.add_edge(
        #     torch.tensor([[idx, closest_base.idx]]), torch.tensor([[1.0]]).float()
        # )

    graph.view()

    # skeleton_bases.pop(0)

    quit()

    skeleton_bases_cld = skeleton_bases_as_cloud(skeletons_sorted)

    while len(skeleton_bases_cld) > 0:
        idx, vector = find_closest_skeleton_base(skeleton, skeleton_bases_cld)

        skeleton_bases_cld = skeleton_bases_cld.delete(idx)

    quit()

    # skeleton = skeletons_sorted.pop(0)

    skeleton_base_idx = [find_skeleton_base(s) for s in tqdm(skeletons_sorted)]

    cld = Cloud(skeleton_base_verts)

    add_base_connection(skeletons_sorted[idx], skeleton_base_idx[idx], vector)

    quit()

    # while len(skeletons_sorted) > 0:
    #    vectors, idxs, radiuses = pts_to_nearest_tube(cld.xyz, skeleton.to_tubes())
    #    closest_skeleton = skeletons_sorted.pop(torch.argmin(radiuses))

    # print(radiuses)

    o3d_viewer([disjoint_skeleton.as_o3d_tube(), cld.as_o3d_cld()])

    find_closest_skeleton_base(skeleton, cld)

    quit()

    while len(skeletons_sorted) != 0:
        idx = find_closest_skeleton_idx(skeleton, skeletons_sorted)
        base_vert_idx = find_skeleton_base(skeletons_sorted.pop(idx))

    # for skeleton in tqdm(skeletons_sorted[1:], desc="Find bases"):
    #     base_idx = find_skeleton_base(skeleton)

    # find_skeleton_base(main_skeleton)

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
