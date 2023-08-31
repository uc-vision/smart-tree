import random
from abc import ABC, abstractmethod
from typing import List, Mapping, Sequence

import numpy as np
import torch
from beartype import beartype
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from taichi_perlin import dropout_3d

from smart_tree.data_types.cloud import Cloud
from smart_tree.util.maths import euler_angles_to_rotation


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
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
        centre, (x, y, z) = cloud.bbox
        return cloud.translate(-centre + torch.tensor([0, y, 0], device=centre.device))


class RandomFlips(Augmentation):
    def __init__(self, x_prob: float = 0.0, y_prob: float = 0.0, z_prob: float = 0.0):
        self.prob = torch.tensor([x_prob, y_prob, z_prob], dtype=torch.float)

    def __call__(self, cloud: Cloud):
        flips = (
            (torch.rand(3) < self.prob)
            .to(dtype=cloud.xyz.dtype, device=cloud.device)
            .view(1, 3)
        )

        return cloud.scale(-flips * 2 + 1)


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
        min_corner = random_pt - (self.size / 2)
        max_corner = random_pt + (self.size / 2)

        mask = torch.logical_and(
            cloud.xyz >= min_corner,
            cloud.xyz <= max_corner,
        ).all(dim=1)

        return cloud.filter(mask)


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


class RandomDropout(Augmentation):
    def __init__(self, max_drop_out):
        self.max_drop_out = max_drop_out

    def __call__(self, cloud: Cloud) -> Cloud:
        num_indices = int(
            (1.0 - (self.max_drop_out * torch.rand(1, device=cloud.xyz.device)))
            * cloud.xyz.shape[0]
        )

        indices = torch.randint(
            high=cloud.xyz.shape[0],
            size=(num_indices, 1),
            device=cloud.xyz.device,
        ).squeeze(1)
        return cloud.filter(indices)


class RandomAugmentation(Augmentation):
    @beartype
    def __init__(
        self, shuffle: bool, apply_prob: float, augmentations: Sequence[Augmentation]
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
    @beartype
    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, cloud):
        for augmentation in self.augmentations:
            cloud = augmentation(cloud)
        return cloud
