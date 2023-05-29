import random

import numpy as np
import torch
import random
import open3d as o3d

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List

from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.util.math.maths import euler_angles_to_rotation
from smart_tree.util.file import load_o3d_mesh
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
        voxel_size: float,
        number_meshes: int,
        min_size: float,
        max_size: float,
    ):
        self.mesh_paths = list(mesh_directory.glob("*"))
        self.voxel_size = voxel_size
        self.number_meshes = number_meshes
        self.min_size = min_size
        self.max_size = max_size
        self.class_number = 3

    def __call__(self, cloud):
        for i in range(self.number_meshes):
            mesh = load_o3d_mesh(
                str(self.mesh_paths[random.randint(0, len(self.mesh_paths))])
            )

            mesh = mesh.scale(0.01, center=mesh.get_center())

            mesh = mesh.translate(-mesh.get_center())

            mesh_pts = mesh.sample_points_uniformly(
                int(1000 * mesh.get_surface_area() / self.voxel_size),
            ).paint_uniform_color(np.random.rand(3))

            lc = LabelledCloud.from_o3d_cld(
                mesh_pts,
                class_l=torch.ones(np.asarray(mesh_pts.points).shape[0])
                * self.class_number,
            )

            cloud += lc

        # load random mesh
        # voxelize mesh
        # create labelled cloud
        # randomly translate / rotate it
        # return new cloud

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
        voxel_size=0.01,
        number_meshes=5,
        min_size=0.01,
        max_size=0.5,
    )

    cld, _ = load_data_npz(Path("/local/_smart-tree/evaluation-data/gt/apple_12.npz"))

    cld = mesh_adder(cld)

    cld.view()

    # o3d_viewer([cld])

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
