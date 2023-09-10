import torch
from typing import List

from smart_tree.data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.util.queries import point_to_point_squared_distances


def find_closest_skeleton(skeleton: TreeSkeleton, query_skeletons: List[TreeSkeleton]):
    min_dist = torch.inf

    for query_skeleton in query_skeletons:
        dist = torch.min(
            point_to_point_squared_distances(skeleton.xyz, query_skeleton.xyz)
        )

        if dist < min_dist:
            min_dist = dist
            closest_skeleton = query_skeleton

    return closest_skeleton


if __name__ == "__main__":
    disjoint_skeleton = DisjointTreeSkeleton.from_pickle(
        "/local/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
    )

    skeletons_sorted = sorted(
        disjoint_skeleton.skeletons,
        key=lambda x: x.length,
        reverse=True,
    )

    main_skeleton = skeletons_sorted[0]

    closest_skeleton = find_closest_skeleton(main_skeleton, skeletons_sorted[1:])

    print(main_skeleton)

    main_skeleton.view()
    closest_skeleton.view()

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
