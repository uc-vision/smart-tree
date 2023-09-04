# test_augmentation.py


import numpy as np
import pytest
import torch

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.augmentations import (
    AugmentationPipeline,
    CentreCloud,
    FixedRotate,
    FixedTranslate,
    RandomAugmentation,
    RandomCrop,
    RandomCubicCrop,
    RandomDropout,
    RandomFlips,
    RandomRotate,
    RandomTranslate,
    Scale,
    VoxelDownsample,
)

# ... (import other necessary modules)


# Fixture to create a sample Cloud object for testing
@pytest.fixture
def sample_cloud():
    xyz = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
    rgb = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
    return Cloud(xyz=xyz, rgb=rgb)


# Test Scale Augmentation
def test_scale_augmentation(sample_cloud):
    min_scale = 0.5
    max_scale = 2.0
    augmentation = Scale(min_scale, max_scale)
    augmented_cloud = augmentation(sample_cloud)
    assert augmented_cloud.xyz.shape == sample_cloud.xyz.shape
    assert torch.all(augmented_cloud.xyz >= min_scale * sample_cloud.xyz)
    assert torch.all(augmented_cloud.xyz <= max_scale * sample_cloud.xyz)


# Test FixedRotate Augmentation
def test_fixed_rotate_augmentation(sample_cloud):
    xyz = [0.1, 0.2, 0.3]
    augmentation = FixedRotate(xyz)
    augmented_cloud = augmentation(sample_cloud)
    # You may add more assertions based on your specific rotation logic


# ... (write test functions for other augmentations)


# Test AugmentationPipeline
def test_augmentation_pipeline(sample_cloud):
    augmentations = [
        Scale(0.5, 2.0),
        RandomRotate(30.0, 45.0, 60.0),
        RandomFlips(0.5, 0.3, 0.2),
    ]
    augmentation = AugmentationPipeline(augmentations)
    augmented_cloud = augmentation(sample_cloud)
    # You may add more assertions based on your specific augmentation pipeline logic


# Additional test cases for augmentations


# Test CentreCloud Augmentation
def test_centre_cloud_augmentation(sample_cloud):
    augmentation = CentreCloud()
    augmented_cloud = augmentation(sample_cloud)
    centered_centroid = torch.mean(augmented_cloud.xyz, dim=0)
    tolerance = 1e-5
    assert torch.all(torch.abs(centered_centroid) < tolerance)


# Test RandomFlips Augmentation
def test_random_flips_augmentation(sample_cloud):
    augmentation = RandomFlips(x_prob=0.5, y_prob=0.5, z_prob=0.5)
    augmented_cloud = augmentation(sample_cloud)
    assert augmented_cloud.xyz.shape == sample_cloud.xyz.shape
    assert torch.all(
        torch.logical_or(
            augmented_cloud.xyz == sample_cloud.xyz,
            augmented_cloud.xyz == -sample_cloud.xyz,
        )
    )


# Test VoxelDownsample Augmentation
def test_voxel_downsample_augmentation(sample_cloud):
    voxel_size = 0.1
    augmentation = VoxelDownsample(voxel_size)
    augmented_cloud = augmentation(sample_cloud)
    # Assert that the number of points in the downsampled cloud is less or equal
    # to the number of points in the original cloud
    assert augmented_cloud.xyz.shape[0] <= sample_cloud.xyz.shape[0]


# Test FixedTranslate Augmentation
def test_fixed_translate_augmentation(sample_cloud):
    xyz = [0.1, -0.2, 0.3]
    augmentation = FixedTranslate(xyz)
    augmented_cloud = augmentation(sample_cloud)
    assert augmented_cloud.xyz.shape == sample_cloud.xyz.shape
    assert torch.all(augmented_cloud.xyz == (sample_cloud.xyz + torch.tensor(xyz)))


# Test RandomCrop Augmentation
def test_random_crop_augmentation(sample_cloud):
    max_x = 0.5
    max_y = 0.5
    max_z = 0.5
    augmentation = RandomCrop(max_x, max_y, max_z)
    augmented_cloud = augmentation(sample_cloud)
    # Check if all points in the augmented cloud are within the crop limits
    assert torch.all(
        torch.logical_and(
            augmented_cloud.xyz >= sample_cloud.min_xyz,
            augmented_cloud.xyz <= sample_cloud.max_xyz,
        ).all(dim=1)
    )


# Test RandomCubicCrop Augmentation
def test_random_cubic_crop_augmentation(sample_cloud):
    size = 0.5
    augmentation = RandomCubicCrop(size)
    augmented_cloud = augmentation(sample_cloud)
    # Check if all points in the augmented cloud are within the cubic crop
    assert torch.all(
        torch.logical_and(
            augmented_cloud.xyz >= sample_cloud.min_xyz - (size / 2),
            augmented_cloud.xyz <= sample_cloud.max_xyz + (size / 2),
        ).all(dim=1)
    )


# Test RandomRotate Augmentation
def test_random_rotate_augmentation(sample_cloud):
    max_x = 30.0
    max_y = 45.0
    max_z = 60.0
    augmentation = RandomRotate(max_x, max_y, max_z)
    augmented_cloud = augmentation(sample_cloud)
    # You may add more assertions based on your specific rotation logic


# Test RandomTranslate Augmentation
def test_random_translate_augmentation(sample_cloud):
    max_x = 0.2
    max_y = 0.2
    max_z = 0.2
    augmentation = RandomTranslate(max_x, max_y, max_z)
    augmented_cloud = augmentation(sample_cloud)
    # You may add more assertions based on your specific translation logic


# Test RandomDropout Augmentation
def test_random_dropout_augmentation(sample_cloud):
    max_drop_out = 0.5
    augmentation = RandomDropout(max_drop_out)
    augmented_cloud = augmentation(sample_cloud)
    # Ensure that the number of points in the augmented cloud is less than or equal
    # to the number of points in the original cloud after dropout
    assert augmented_cloud.xyz.shape[0] <= sample_cloud.xyz.shape[0]


# Test RandomAugmentation
def test_random_augmentation(sample_cloud):
    augmentations = [
        Scale(0.5, 2.0),
        RandomRotate(30.0, 45.0, 60.0),
        RandomFlips(0.5, 0.3, 0.2),
    ]
    augmentation = RandomAugmentation(
        shuffle=True, apply_prob=0.8, augmentations=augmentations
    )
    augmented_cloud = augmentation(sample_cloud)
    # You may add more assertions based on your specific random
