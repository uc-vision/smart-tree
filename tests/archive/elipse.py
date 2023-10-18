from pathlib import Path

import numpy as np
from tqdm import tqdm

from smart_tree.o3d_abstractions.geometries import o3d_elipsoid, o3d_merge_meshes
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.util.file import load_cloud
from smart_tree.util.maths import rotation_matrix_from_vectors_np

if __name__ == "__main__":
    cld = load_cloud(Path("/local/UC-10/npz_4mm/london_10.npz"))

    centres = cld.medial_pts[::200]
    directions = cld.branch_direction[::200]

    radii = cld.radius[::200]

    o3d_elipses = []

    for c, d, r in tqdm(zip(centres, directions, radii)):
        elipse_params = r.repeat(3) + np.array(
            [np.sqrt(r) / 5, 0, 0]
        )  # (np.abs(d) * r)  # r.repeat(3)  # + (d * r)

        rot_mat = rotation_matrix_from_vectors_np(np.array([1.0, 0.0, 0.0]), d)

        o3d_elipses.append(
            o3d_elipsoid(elipse_params.numpy(), c.numpy()).rotate(rot_mat)
        )

    o3d_viewer([o3d_merge_meshes(o3d_elipses), cld.to_o3d_cld()])

    print(centres)

    quit()

    cld.medial_pts

    radius = np.array([0.1])

    direction = np.array([1, 0.0, 0.0])

    elipse_params = radius.repeat(3) + (direction * radius)

    elipse = o3d_elipsoid(elipse_params)

    o3d_viewer([elipse])
