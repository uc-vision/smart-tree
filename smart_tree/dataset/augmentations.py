import random
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from beartype import beartype
from taichi_perlin import dropout_3d

from ..data_types.cloud import Cloud
from ..util.maths import euler_angles_to_rotation


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale: float, max_scale: float):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, cloud):
        log_min, log_max = np.log(self.min_scale), np.log(self.max_scale)

        t = (
            torch.rand(1, device=cloud.xyz.device, dtype=cloud.xyz.dtype)
            * (log_max - log_min)
            + log_min
        )
        return cloud.scale(torch.exp(t))


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


class VoxelDownsample(Augmentation):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.voxel_downsample(self.voxel_size)


class FixedTranslate(Augmentation):
    def __init__(self, xyz):
        self.xyz = torch.tensor(xyz)

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.translate(self.xyz)


class RandomCrop(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_translation = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        offset = (
            torch.rand(3, device=cloud.xyz.device) - 0.5
        ) * self.max_translation.to(device=cloud.xyz.device)

        mask = torch.logical_and(
            cloud.xyz + offset >= cloud.min_xyz,
            cloud.xyz + offset <= cloud.max_xyz,
        ).all(dim=1)

        return cloud.filter(mask)


class RandomCubicCrop(Augmentation):
    def __init__(self, size):
        self.size = size

    def __call__(self, cloud):
        random_pt = cloud.xyz[torch.randint(0, cloud.xyz.shape[0], (1,))]
        min_corner = random_pt - (self.size / 2)
        max_corner = random_pt + (self.size / 2)

        mask = torch.logical_and(
            cloud.xyz >= min_corner,
            cloud.xyz <= max_corner,
        ).all(dim=1)

        return cloud.filter(mask)



class RandomGaussianNoise(Augmentation):
    def __init__(self, mean=0.0, std=0.01, prob=0.0, magnitude=1.0):
        self.mean = mean
        self.std = std
        self.prob = prob
        self.magnitude = magnitude

    def __call__(self, cloud):
        mask = torch.rand(cloud.xyz.shape[0]) < self.prob
        noise = (
            torch.randn(cloud.xyz.shape, device=cloud.xyz.device) * self.std + self.mean
        ).float()
        noise *= self.magnitude
        cloud.xyz[mask] += noise[mask]
        return cloud.with_xyz(cloud.xyz)


class RandomRotate(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_rots = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        x, y, z = (torch.rand(3) - 0.5) * self.max_rots

        self.rot_mat = euler_angles_to_rotation(torch.tensor([x, y, z])).to(
            device=cloud.xyz.device
        )
        return cloud.rotate(self.rot_mat.float())


class RandomTranslate(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_translation = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        return cloud.translate(
            (torch.rand(3, device=cloud.xyz.device) - 0.5)
            * self.max_translation.to(device=cloud.xyz.device)
        )


class Dropout3D(Augmentation):
    def __init__(self, **kwargs):
        kwargs = {
            k: tuple(v) if isinstance(v, Sequence) else v for k, v in kwargs.items()
        }

        self.params = dropout_3d.DropoutParams(**kwargs)
        self.dropout = dropout_3d.PointDropout(self.params)

    def __call__(self, cloud):
        points, mask = self.dropout(cloud.xyz)
        return cloud.filter(mask).with_xyz(points)


class RandomDropout:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, cloud: Cloud) -> Cloud:
        mask = torch.rand(cloud.xyz.shape[0]) < self.probability
        return cloud.filter(~mask)



class RandomAugmentation(Augmentation):
    def __init__(
        self,
        shuffle: bool,
        apply_prob: float,
        augmentations: Sequence[Augmentation],
    ):
        self.augmentations = augmentations

        self.shuffle = shuffle
        self.apply_prob = apply_prob

    def __call__(self, cloud):
        randomized = list(self.augmentations)
        if self.shuffle:
            random.shuffle(randomized)

        for augmentation in randomized:
            if torch.rand(1) < self.apply_prob:
                cloud = augmentation(cloud)

        return cloud


class AugmentationPipeline(Augmentation):
    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, cloud):
        for augmentation in self.augmentations:
            cloud = augmentation(cloud)
        return cloud
