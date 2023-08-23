import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path

from smart_tree.data_types.cloud import Cloud
from smart_tree.util.math.maths import euler_angles_to_rotation


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, cloud: Cloud) -> Cloud:
        t = torch.randn(1, device=cloud.xyz.device) * (self.max_scale - self.min_scale)
        return cloud.scale(t + self.min_scale)


class FixedRotate(Augmentation):
    def __init__(self, xyz):
        self.xyz = xyz

    def __call__(self, cloud: Cloud) -> Cloud:
        self.rot_mat = euler_angles_to_rotation(
            torch.tensor(self.xyz), device=cloud.device
        ).float()
        return cloud.rotate(self.rot_mat)


class CentreCloud(Augmentation):
    def __call__(self, cloud: Cloud) -> Cloud:
        centre, (x, y, z) = cloud.bbox
        return cloud.translate(-centre + torch.tensor([0, y, 0]))


class VoxelDownsample(Augmentation):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.voxel_down_sample(self.voxel_size)


class FixedTranslate(Augmentation):
    def __init__(self, xyz):
        self.xyz = torch.tensor(xyz)

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.translate(self.xyz)


class RandomDropout(Augmentation):
    def __init__(self, max_drop_out):
        self.max_drop_out = max_drop_out

    def __call__(self, cloud: Cloud) -> Cloud:
        num_indices = int(
            (1.0 - (self.max_drop_out * torch.rand(1, device=cloud.xyz.device)))
            * cloud.xyz.shape[0]
        )

        indices = torch.randint(
            high=cloud.xyz.shape[0], size=(num_indices, 1), device=cloud.xyz.device
        ).squeeze(1)
        return cloud.filter(indices)


class AugmentationPipeline:
    def __init__(self, augmentation_fns: List[Augmentation] = []):
        self.pipeline = augmentation_fns

    def __call__(self, cloud: Cloud) -> Cloud:
        for augmentation in self.pipeline:
            cloud = augmentation(cloud)
        return cloud

    @staticmethod
    def from_cfg(cfg):
        if cfg == None:
            return AugmentationPipeline([])
        return AugmentationPipeline([(cfg[key]) for key in cfg.keys()])
