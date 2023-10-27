import open3d as o3d
import numpy as np

def split_by_color(pcd):
    """
    Splits a point cloud by unique colors.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud

    Returns:
        dict: Dictionary where keys are unique colors and values are point clouds with those colors
    """
    colors = np.asarray(pcd.colors)
    unique_colors, indices = np.unique(colors, axis=0, return_inverse=True)

    color_to_pcd = {}
    for color_index, unique_color in enumerate(unique_colors):
        mask = indices == color_index
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        sub_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        color_to_pcd[tuple(unique_color)] = sub_pcd
    return color_to_pcd


if __name__ == "__main__":
    # Read point cloud
    pcd = o3d.io.read_point_cloud("/local/downloads/segmented.ply")

    # Split point cloud by color
    color_to_pcd_map = split_by_color(pcd)

    # Example: Save each separate color point cloud
    for color, sub_pcd in color_to_pcd_map.items():
        color_str = "_".join(map(str, [int(c * 255) for c in color]))
        o3d.io.write_point_cloud(f"/local/downloads/split_{color_str}.ply", sub_pcd)