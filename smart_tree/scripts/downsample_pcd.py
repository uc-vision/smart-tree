import open3d as o3d

if __name__ == "__main__":
    cld = o3d.io.read_point_cloud(
        "/home/harry/Desktop/scion_uc_collab/nerf_cleaned.ply",
    ).voxel_down_sample(0.001)
    o3d.io.write_point_cloud(
        "/home/harry/Desktop/scion_uc_collab/nerf_cleaned_0.001.ply", cld
    )
