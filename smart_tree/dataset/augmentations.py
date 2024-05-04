from abc import ABC, abstractmethod
from typing import Sequence

import torch
from beartype import beartype

from ..data_types.cloud import Cloud
from ..util.maths import euler_angles_to_rotation
from ..skeleton.filter import outlier_removal


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, cloud: Cloud) -> Cloud:
        t = torch.rand(1, device=cloud.xyz.device) * (self.max_scale - self.min_scale)
        return cloud.scale(t + self.min_scale)


class FixedRotate(Augmentation):
    def __init__(self, xyz):
        self.xyz = xyz

    def __call__(self, cloud: Cloud) -> Cloud:
        self.rot_mat = euler_angles_to_rotation(
            torch.tensor(self.xyz, device=cloud.xyz.device)
        ).float()
        return cloud.rotate(self.rot_mat)


class CentreCloud(Augmentation):
    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.translate(-cloud.centre)


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


class RadiusOutlierRemoval(Augmentation):
    def __init__(self, number_points: int = 8, radius: float = 0.025):
        self.number_points = number_points
        self.radius = torch.tensor(radius)

    def __call__(self, cloud: Cloud) -> Cloud:

        mask = outlier_removal(
            cloud.xyz, self.radius.to(cloud.device), nb_points=self.number_points
        )

        return cloud.filter(mask)


class RandomCrop(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_translation = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        offset = (
            torch.rand(3, device=cloud.xyz.device) - 0.5
        ) * self.max_translation.to(device=cloud.xyz.device)

        p = cloud.xyz + offset
        mask = torch.logical_and(p >= cloud.min_xyz, p <= cloud.max_xyz).all(dim=1)

        return cloud.filter(mask)


class RandomCubicCrop(Augmentation):
    def __init__(self, size):
        self.size = size

    def __call__(self, cloud):
        random_pt = cloud.xyz[torch.randint(0, cloud.xyz.shape[0], (1,))]
        min_corner = random_pt - self.size / 2
        max_corner = random_pt + self.size / 2

        mask = torch.logical_and(
            cloud.xyz >= min_corner,
            cloud.xyz <= max_corner,
        ).all(dim=1)

        return cloud.filter(mask)


class FilterByClass(Augmentation):
    def __init__(self, classes: list):
        self.classes = torch.tensor(classes)

    def __call__(self, cloud):
        mask = torch.isin(cloud.class_l, self.classes.to(cloud.device))
        return cloud.filter(mask.view(-1))


class RandomTranslate(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_translation = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        return cloud.translate(
            (torch.rand(3, device=cloud.xyz.device) - 0.5)
            * self.max_translation.to(device=cloud.xyz.device)
        )


class RandomDropout(Augmentation):
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, cloud: Cloud) -> Cloud:
        mask = torch.rand(cloud.xyz.shape[0]) < self.prob
        return cloud.filter(~mask)


class RandomFlips(Augmentation):
    def __init__(self, x_prob: float = 0.0, y_prob: float = 0.0, z_prob: float = 0.0):
        self.prob = torch.tensor([x_prob, y_prob, z_prob], dtype=torch.float)

    def __call__(self, cloud: Cloud):
        flips = (
            (torch.rand(3) < self.prob)
            .to(dtype=cloud.xyz.dtype, device=cloud.device)
            .view(1, 3)
        )
        return cloud.scale(-flips * 2.0 + 1.0)


class RandomGaussianNoise(Augmentation):
    def __init__(self, mean=0.0, std=0.01, prob=0.0, magnitude=1.0):
        self.mean = mean
        self.std = std
        self.prob = prob
        self.magnitude = magnitude

    def __call__(self, cloud):
        mask = torch.rand(cloud.xyz.shape[0]) < self.prob
        noise = (
            torch.randn(cloud.xyz.shape, device=cloud.xyz.device, dtype=cloud.xyz.dtype)
            * self.std
            + self.mean
        )
        noise *= self.magnitude
        cloud.xyz[mask] += noise[mask].to(cloud.xyz.dtype)
        return cloud.with_xyz(cloud.xyz)


class AugmentationPipeline(Augmentation):
    @beartype
    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, cloud):
        for augmentation in self.augmentations:
            cloud = augmentation(cloud)
        return cloud