from pathlib import Path

import pytest
import torch

from smart_tree.data_types.cloud import Cloud, LabelledCloud


@pytest.fixture
def sample_cloud_data():
    xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rgb = torch.tensor([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]])
    filename = Path("sample_cloud.txt")
    return xyz, rgb, filename


def test_len_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)
    assert len(cloud) == 2


def test_str_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)
    expected_str = (
        f"{'*' * 80}"
        f"Cloud:\n"
        f"Coloured: True\n"
        f"Filename: {filename}\n"
        f"{'*' * 80}"
    )
    assert str(cloud) == expected_str


def test_scale_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)
    factor = 2.0
    scaled_cloud = cloud.scale(factor)

    assert torch.allclose(scaled_cloud.xyz, xyz * factor)
    assert torch.equal(scaled_cloud.rgb, rgb)
    assert scaled_cloud.filename == filename


def test_translate_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    translation_vector = torch.tensor([1.0, -2.0, 3.0])
    translated_cloud = cloud.translate(translation_vector)

    expected_xyz = xyz + translation_vector
    assert torch.allclose(translated_cloud.xyz, expected_xyz)
    assert torch.equal(translated_cloud.rgb, rgb)
    assert translated_cloud.filename == filename


def test_rotate_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_cloud = cloud.rotate(rotation_matrix)

    expected_xyz = torch.matmul(rotation_matrix, xyz.T).T
    assert torch.allclose(rotated_cloud.xyz, expected_xyz)
    assert torch.equal(rotated_cloud.rgb, rgb)
    assert rotated_cloud.filename == filename


def test_filter_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    mask = torch.tensor([True, False])
    filtered_cloud = cloud.filter(mask)

    expected_xyz = xyz[mask]
    expected_rgb = rgb[mask]
    assert torch.allclose(filtered_cloud.xyz, expected_xyz)
    assert torch.equal(filtered_cloud.rgb, expected_rgb)
    assert filtered_cloud.filename == filename


def test_to_device_method(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_on_device = cloud.to_device(device)

    assert cloud_on_device.xyz.device == device
    if cloud.rgb is not None:
        assert cloud_on_device.rgb.device == device
    assert cloud_on_device.filename == filename


def test_device_property(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    assert cloud.device == xyz.device
    if cloud.rgb is not None:
        assert cloud.device == cloud.rgb.device


def test_bounding_box_property(sample_cloud_data):
    xyz, rgb, filename = sample_cloud_data
    cloud = Cloud(xyz=xyz, rgb=rgb, filename=filename)

    center, dimensions = cloud.bounding_box

    min_xyz = xyz.min(dim=0).values
    max_xyz = xyz.max(dim=0).values
    expected_center = (min_xyz + max_xyz) / 2
    expected_dimensions = (max_xyz - min_xyz) / 2

    assert torch.allclose(center, expected_center)
    assert torch.allclose(dimensions, expected_dimensions)


@pytest.fixture
def sample_labelled_cloud():
    xyz = torch.randn(5, 3, dtype=torch.float32)
    rgb = torch.randn(5, 3, dtype=torch.float32)
    medial_vector = torch.randn(5, 3, dtype=torch.float32)
    branch_direction = torch.randn(5, 3, dtype=torch.float32)
    branch_ids = torch.randn(5, 1, dtype=torch.float32)
    class_l = torch.randn(5, 1, dtype=torch.float32)

    return LabelledCloud(
        xyz=xyz,
        rgb=rgb,
        medial_vector=medial_vector,
        branch_direction=branch_direction,
        branch_ids=branch_ids,
        class_l=class_l,
    )


def test_scale_method(sample_labelled_cloud):
    factor = 2.0
    scaled_cloud = sample_labelled_cloud.scale(factor)

    # Check that the xyz, medial_vector, branch_direction, and branch_ids are scaled correctly
    assert torch.allclose(scaled_cloud.xyz, sample_labelled_cloud.xyz * factor)
    assert torch.allclose(
        scaled_cloud.medial_vector, sample_labelled_cloud.medial_vector * factor
    )
    assert torch.allclose(
        scaled_cloud.branch_direction, sample_labelled_cloud.branch_direction
    )
    assert torch.allclose(scaled_cloud.branch_ids, sample_labelled_cloud.branch_ids)

    # Check that other attributes remain unchanged
    assert torch.equal(scaled_cloud.rgb, sample_labelled_cloud.rgb)
    assert torch.equal(scaled_cloud.class_l, sample_labelled_cloud.class_l)


def test_rotate_method(sample_labelled_cloud):
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_cloud = sample_labelled_cloud.rotate(rotation_matrix)

    # Check that the xyz, medial_vector, and branch_direction are rotated correctly
    assert torch.allclose(
        rotated_cloud.xyz, torch.matmul(sample_labelled_cloud.xyz, rotation_matrix.T)
    )
    assert torch.allclose(
        rotated_cloud.medial_vector,
        torch.matmul(sample_labelled_cloud.medial_vector, rotation_matrix.T),
    )
    assert torch.allclose(
        rotated_cloud.branch_direction,
        torch.matmul(sample_labelled_cloud.branch_direction, rotation_matrix.T),
    )

    # Check that other attributes remain unchanged
    assert torch.equal(rotated_cloud.rgb, sample_labelled_cloud.rgb)
    assert torch.equal(rotated_cloud.branch_ids, sample_labelled_cloud.branch_ids)
    assert torch.equal(rotated_cloud.class_l, sample_labelled_cloud.class_l)


@pytest.fixture
def sample_labelled_cloud_no_optionals():
    xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return LabelledCloud(xyz=xyz)


# Define a fixture for a sample LabelledCloud instance with some optional properties
@pytest.fixture
def sample_labelled_cloud_with_optionals():
    xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    medial_vector = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    class_l = torch.tensor([[0], [1]], dtype=torch.float32)
    return LabelledCloud(xyz=xyz, medial_vector=medial_vector, class_l=class_l)


def test_scale_method_no_optionals(sample_labelled_cloud_no_optionals):
    factor = 2.0
    scaled_cloud = sample_labelled_cloud_no_optionals.scale(factor)

    # Check that the xyz is scaled correctly
    assert torch.allclose(
        scaled_cloud.xyz, sample_labelled_cloud_no_optionals.xyz * factor
    )

    # Check that the optional properties remain None
    assert scaled_cloud.medial_vector is None
    assert scaled_cloud.branch_direction is None
    assert scaled_cloud.branch_ids is None
    assert scaled_cloud.class_l is None
    assert scaled_cloud.rgb is None


def test_rotate_method_no_optionals(sample_labelled_cloud_no_optionals):
    rotation_matrix = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    rotated_cloud = sample_labelled_cloud_no_optionals.rotate(rotation_matrix)

    # Check that the xyz is rotated correctly
    assert torch.allclose(
        rotated_cloud.xyz,
        torch.matmul(sample_labelled_cloud_no_optionals.xyz, rotation_matrix.T),
    )

    # Check that the optional properties remain None
    assert rotated_cloud.medial_vector is None
    assert rotated_cloud.branch_direction is None
    assert rotated_cloud.branch_ids is None
    assert rotated_cloud.class_l is None
    assert rotated_cloud.rgb is None


def test_scale_method_with_optionals(sample_labelled_cloud_with_optionals):
    factor = 2.0
    scaled_cloud = sample_labelled_cloud_with_optionals.scale(factor)

    # Check that the xyz and optional properties are scaled correctly
    assert torch.allclose(
        scaled_cloud.xyz, sample_labelled_cloud_with_optionals.xyz * factor
    )
    assert torch.allclose(
        scaled_cloud.medial_vector,
        sample_labelled_cloud_with_optionals.medial_vector * factor,
    )
    assert torch.allclose(
        scaled_cloud.class_l, sample_labelled_cloud_with_optionals.class_l
    )

    # Check that the optional properties remain None
    assert scaled_cloud.branch_direction is None
    assert scaled_cloud.branch_ids is None
    assert scaled_cloud.rgb is None


def test_rotate_method_with_optionals(sample_labelled_cloud_with_optionals):
    rotation_matrix = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    rotated_cloud = sample_labelled_cloud_with_optionals.rotate(rotation_matrix)

    # Check that the xyz and optional properties are rotated correctly
    assert torch.allclose(
        rotated_cloud.xyz,
        torch.matmul(sample_labelled_cloud_with_optionals.xyz, rotation_matrix.T),
    )
    assert torch.allclose(
        rotated_cloud.medial_vector,
        torch.matmul(
            sample_labelled_cloud_with_optionals.medial_vector, rotation_matrix.T
        ),
    )
    assert torch.allclose(
        rotated_cloud.class_l, sample_labelled_cloud_with_optionals.class_l
    )

    # Check that the optional properties remain None
    assert rotated_cloud.branch_direction is None
    assert rotated_cloud.branch_ids is None
    assert rotated_cloud.rgb is None


def test_pin_memory_method(sample_labelled_cloud):
    # Ensure that the 'pin_memory' method correctly pins float tensor properties
    pinned_cloud = sample_labelled_cloud.pin_memory()

    # Check if 'xyz', 'rgb', 'medial_vector', 'branch_direction', 'branch_ids', and 'class_l' are pinned
    assert all(
        isinstance(getattr(pinned_cloud, prop_name), torch.Tensor)
        for prop_name in [
            "xyz",
            "rgb",
            "medial_vector",
            "branch_direction",
            "branch_ids",
            "class_l",
        ]
    )


def test_to_device_method(sample_labelled_cloud):
    # Create a target device
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure that the 'to_device' method correctly moves float tensor properties to the target device
    device_cloud = sample_labelled_cloud.to_device(target_device)

    # Check if 'xyz', 'rgb', 'medial_vector', 'branch_direction', 'branch_ids', and 'class_l' are on the target device
    assert all(
        getattr(device_cloud, prop_name).device == target_device
        and getattr(device_cloud, prop_name).dtype in [torch.float32, torch.float64]
        for prop_name in [
            "xyz",
            "rgb",
            "medial_vector",
            "branch_direction",
            "branch_ids",
            "class_l",
        ]
    )


def test_filter_method(sample_labelled_cloud):
    # Create a mask for filtering
    mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)

    # Apply the 'filter' method to filter the cloud based on the mask
    filtered_cloud = sample_labelled_cloud.filter(mask)

    # Check if 'xyz', 'rgb', 'medial_vector', 'branch_direction', 'branch_ids', and 'class_l' are filtered correctly
    assert torch.equal(filtered_cloud.xyz, sample_labelled_cloud.xyz[mask])
    assert torch.equal(filtered_cloud.rgb, sample_labelled_cloud.rgb[mask])
    assert torch.equal(
        filtered_cloud.medial_vector, sample_labelled_cloud.medial_vector[mask]
    )
    assert torch.equal(
        filtered_cloud.branch_direction, sample_labelled_cloud.branch_direction[mask]
    )
    assert torch.equal(
        filtered_cloud.branch_ids, sample_labelled_cloud.branch_ids[mask]
    )
    assert torch.equal(filtered_cloud.class_l, sample_labelled_cloud.class_l[mask])
