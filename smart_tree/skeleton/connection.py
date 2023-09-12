from typing import List

import torch
from torch.nn import functional as F
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.skeleton.shortest_path import shortest_paths
from smart_tree.util.maths import torch_dot
from smart_tree.util.queries import skeleton_to_points, skeleton_to_skeleton_distance


def find_closest_skeleton_idx(
    skeleton: TreeSkeleton,
    query_skeletons: List[TreeSkeleton],
) -> int:
    min_dist = torch.inf

    for idx, query_skeleton in tqdm(enumerate(query_skeletons)):
        dist = torch.min(skeleton_to_skeleton_distance(skeleton, query_skeleton))

        if dist < min_dist:
            min_dist = dist
            closest_skeleton_idx = idx

    return closest_skeleton_idx


def find_skeleton_base(skeleton: TreeSkeleton):
    graph = skeleton.to_graph()

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
        torch_dot(path_direction, -branch_direction).clamp(min=1e-15) ** 2
    )

    verts, preds, distance = shortest_paths(
        start_idx,
        graph.edges,
        new_edge_weights,
        renumber=False,
    )

    return distance.argmax()


def rectify_skeleton(skeleton: TreeSkeleton):
    # we want it as a graph format at some point either start or end?

    skeleton_base_idx = find_skeleton_base(skeleton)
    graph = skeleton.to_graph()

    verts, preds, distance = shortest_paths(
        skeleton_base_idx,
        graph.edges,
        graph.edge_weights,
        renumber=False,
    )


def find_closest_skeleton_base(skeleton: TreeSkeleton, cld: Cloud):
    distances, radii, vectors_ = skeleton_to_points(cld, skeleton)


if __name__ == "__main__":
    print("loading")
    disjoint_skeleton = DisjointTreeSkeleton.from_pickle(
        "/mnt/harry/PhD/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
    )
    print("loaded")

    print("sorting")
    skeletons_sorted = sorted(
        disjoint_skeleton.skeletons,
        key=lambda x: x.length,
        reverse=True,
    )
    print("sorted")

    # tree = merge_trees_in_list(skeletons_sorted)

    # tree.repair()
    # tree.view()

    skeleton = skeletons_sorted.pop(0)

    print("finding bases")
    skeleton_base_idx = [find_skeleton_base(s) for s in tqdm(skeletons_sorted)]
    print("found bases")

    skeleton_base_verts = torch.stack(
        [skel.xyz[i] for skel, i in zip(skeletons_sorted, skeleton_base_idx)]
    ).cpu()

    cld = Cloud(skeleton_base_verts)

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
