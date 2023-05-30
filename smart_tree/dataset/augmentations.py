import os
import random

import numpy as np
import torch
import random
import open3d as o3d

from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List

from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.util.math.maths import euler_angles_to_rotation
from smart_tree.util.file import (
    load_o3d_mesh,
    save_o3d_mesh,
    save_o3d_cloud,
    load_o3d_cloud,
)
from smart_tree.util.mesh.geometries import o3d_cloud, o3d_sphere
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, cloud):
        t = torch.randn(1, device=cloud.xyz.device) * (self.max_scale - self.min_scale)
        return cloud.scale(t + self.min_scale)


class FixedRotate(Augmentation):
    def __init__(self, xyz):
        self.xyz = xyz

    def __call__(self, cloud):
        self.rot_mat = euler_angles_to_rotation(
            torch.tensor(self.xyz), device=cloud.xyz.device
        ).float()
        return cloud.rotate(self.rot_mat)


class CentreCloud(Augmentation):
    def __call__(self, cloud):
        centre, (x, y, z) = cloud.bbox
        return cloud.translate(-centre + torch.tensor([0, y, 0]))


class VoxelDownsample(Augmentation):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, cloud):
        return cloud.voxel_down_sample(self.voxel_size)


class FixedTranslate(Augmentation):
    def __init__(self, xyz):
        self.xyz = torch.tensor(xyz)

    def __call__(self, cloud):
        return cloud.translate(self.xyz)


class RandomTranslate(Augmentation):
    def __init__(self, std):
        self.std = torch.tensor(std)

    def __call__(self, cloud):
        return cloud.translate(
            torch.normal(
                torch.zeros(3, device=cloud.xyz.device),
                std=self.std.to(cloud.xyz.device),
            )
        )


class RandomMesh(Augmentation):
    def __init__(
        self,
        mesh_directory: Path,
        preprocessed_path: Path,
        voxel_size: float,
        number_meshes: int,
        min_size: float,
        max_size: float,
        max_pts: int,
    ):
        """We want to preprocess the meshes by getting them to the right scale,
        which is done by first converting them from mm to metres, we then
        scale them up to the max_size and then do a point sample, based on target voxel_size
        (ensure we have enough point density at the max size), then revert
        the scale back normal scale in metres. During inference we randomly scale based on the
        min_size and max_size and then translate the points and merge with the input cloud
        """

        self.voxel_size = voxel_size
        self.number_meshes = number_meshes
        self.min_size = min_size
        self.max_size = max_size
        self.class_number = 2
        self.preprocessed_path = preprocessed_path

        if not (os.path.exists(preprocessed_path)):
            os.makedirs(preprocessed_path)

        for mesh_path in tqdm(
            Path(mesh_directory).glob("*.stl"),
            desc="Preprocessing Meshes",
            leave=False,
        ):
            if os.path.isfile(f"{preprocessed_path}/{mesh_path.stem}.pcd"):
                continue
            try:
                mesh = load_o3d_mesh(str(mesh_path))
                pcd = (
                    mesh.scale(0.001, mesh.get_center())
                    .translate(-mesh.get_center())
                    .paint_uniform_color(np.random.rand(3))
                    .scale(max_size, mesh.get_center())
                    .sample_points_uniformly(
                        min(
                            max(int(mesh.get_surface_area() / (voxel_size**2)), 10),
                            max_pts,
                        )
                    )
                    .scale(1 / max_size, mesh.get_center())
                )
                save_o3d_cloud(f"{preprocessed_path}/{mesh_path.stem}.pcd", pcd)
            except:
                print(f"Cloud Generation Failed on {mesh_path}")

    def __call__(self, cloud):
        centre, dimensions = cloud.bbox

        for i in range(self.number_meshes):
            random_pcd_path = random.choice(list(self.preprocessed_path.glob("*.pcd")))
            pcd = load_o3d_cloud(str(random_pcd_path))
            scaled_pcd = pcd.scale(
                random.uniform(self.min_size, self.max_size), pcd.get_center()
            )

            lc = LabelledCloud.from_o3d_cld(
                pcd,
                class_l=torch.ones(np.asarray(pcd.points).shape[0]) * self.class_number,
            )

            lc = lc.rotate(
                euler_angles_to_rotation(torch.rand(3) * torch.pi * 2).to(
                    cloud.xyz.device
                )
            )
            lc = lc.translate(cloud.min_xyz - lc.centre)
            lc = lc.translate(dimensions * torch.rand(3))

            cloud += lc

        return cloud


class RandomDropout(Augmentation):
    def __init__(self, max_drop_out):
        self.max_drop_out = max_drop_out

    def __call__(self, cloud):
        num_indices = int(
            (1.0 - (self.max_drop_out * torch.rand(1, device=cloud.xyz.device)))
            * cloud.xyz.shape[0]
        )

        indices = torch.randint(
            high=cloud.xyz.shape[0], size=(num_indices, 1), device=cloud.xyz.device
        ).squeeze(1)
        return cloud.filter(indices)


class RandomColourDropout(Augmentation):
    def __init__(self, max_drop_out):
        self.max_drop_out = max_drop_out

    def __call__(self, cloud):
        num_indices = int(
            (1.0 - (self.max_drop_out * torch.rand(1, device=cloud.rgb.device)))
            * cloud.xyz.shape[0]
        )

        indices = torch.randint(
            high=cloud.rgb.shape[0], size=(num_indices, 1), device=cloud.rgb.device
        ).squeeze(1)

        cloud.rgb[indices] = torch.ones_like(cloud.rgb[indices])

        return cloud


class AugmentationPipeline:
    def __init__(self, augmentation_fns: List[Augmentation]):
        # config is a dict
        self.pipeline = augmentation_fns

    def __call__(self, cloud):
        for augmentation in self.pipeline:
            cloud = augmentation(cloud)
        return cloud

    @staticmethod
    def from_cfg(cfg):
        if cfg == None:
            return AugmentationPipeline([])
        return AugmentationPipeline([(cfg[key]) for key in cfg.keys()])


if __name__ == "__main__":
    from pathlib import Path
    from smart_tree.util.file import load_data_npz
    from smart_tree.util.visualizer.view import o3d_viewer

    mesh_adder = RandomMesh(
        mesh_directory=Path("/local/Datasets/Thingi10K/raw_meshes/"),
        preprocessed_path=Path(
            "/local/uc-vision/smart-tree/data/things10K_sampled_1mm/"
        ),
        voxel_size=0.001,
        number_meshes=20,
        min_size=0.01,
        max_size=20,
        max_pts=50000,
    )

    cld, _ = load_data_npz(Path("/local/_smart-tree/evaluation-data/gt/apple_12.npz"))

    cld = mesh_adder(cld)

    cld.view()

    o3d_viewer(
        [
            cld.to_o3d_cld(),
            o3d_sphere(cld.min_xyz, radius=0.1, colour=(0, 1, 0)),
            o3d_sphere(cld.max_xyz, radius=0.1),
        ]
    )

    quit()
    centre = CentreCloud()
    rotater = FixedRotate(torch.tensor([torch.pi / 2, torch.pi / 2, torch.pi * 2]))
    do = RandomDropout(0.5)

    centrecld = centre(cld)

    rot_cloud = rotater(centrecld)

    do_cloud = do(cld)
    o3d_viewer(
        [
            cld.to_o3d_cld(),
            centrecld.to_o3d_cld(),
            rot_cloud.to_o3d_cld(),
            do_cloud.to_o3d_cld(),
        ]
    )

    print(cld)

    print(do_cloud)

    # cld.view()

    pass
