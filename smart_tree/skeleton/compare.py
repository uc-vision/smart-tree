from smart_tree.data_types.tree import DisjointTreeSkeleton
from smart_tree.o3d_abstractions.visualizer import o3d_viewer

scion_skeleton = DisjointTreeSkeleton.from_pickle(
    "/mnt/harry/PhD/smart-tree/data/scion.pkl"
)

nerf_skeleton = DisjointTreeSkeleton.from_pickle(
    "/mnt/harry/PhD/smart-tree/data/nerf.pkl"
)


scion_skeleton.repair()
# scion_skeleton.view()

nerf_skeleton.repair()
# scion_skeleton.view()

o3d_viewer(
    [scion_skeleton.as_o3d_lineset(), nerf_skeleton.as_o3d_lineset()],
    line_width=5,
)
