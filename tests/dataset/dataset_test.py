import os
from pathlib import Path

import pytest
import torch

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import Dataset

path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.dirname(path_to_current_file)


# Define test data and paths
json_path = f"{current_directory}/../data/fake-split.json"
directory = f"{current_directory}/../data/"
mode = "train"
# Define other necessary parameters for the Dataset constructor


@pytest.fixture
def dataset():
    return Dataset(json_path, directory, mode)


def test_dataset_initialization(dataset):
    assert dataset.mode == mode
    assert dataset.directory == directory
    # Add more assertions for other attributes


def test_load_cloud(dataset):
    # Create test cloud data and save it to a temporary file
    # Use the temporary file path for testing
    test_cloud_path = Path(f"{directory}/test.npz")
    test_cloud = dataset.load(test_cloud_path)

    # Add assertions to check the loaded cloud object
    assert test_cloud.xyz is not None
    assert test_cloud.rgb is not None  # If applicable
    assert test_cloud.filename == test_cloud_path


def test_process_cloud(dataset):
    test_cloud = Cloud(xyz=torch.tensor([[1.0, 2.0, 3.0]]), rgb=None, filename=None)
    processed_cloud = dataset.process_cloud(test_cloud, "test_cloud.ply")

    assert processed_cloud is not None


if __name__ == "__main__":
    pytest.main()
