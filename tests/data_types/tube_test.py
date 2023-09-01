from dataclasses import asdict
from typing import List

import pytest
import torch

from smart_tree.data_types.tube import CollatedTube, Tube, collate_tubes


# Create a fixture for a sample Tube instance
@pytest.fixture
def sample_tube():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    r1 = torch.tensor([0.5], dtype=torch.float32)
    r2 = torch.tensor([0.7], dtype=torch.float32)
    return Tube(a, b, r1, r2)


# Test the 'to_device' method of the Tube class
def test_tube_to_device(sample_tube):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_tube = sample_tube.to_device(device)

    # Check if the properties are on the target device
    assert device_tube.a.device == device
    assert device_tube.b.device == device
    assert device_tube.r1.device == device
    assert device_tube.r2.device == device


# Create a fixture for a list of sample Tube instances
@pytest.fixture
def sample_tube_list():
    tubes = [
        Tube(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor([0.5]),
            torch.tensor([0.7]),
        ),
        Tube(
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0]),
            torch.tensor([0.6]),
            torch.tensor([0.8]),
        ),
    ]
    return tubes


# Test the 'collate_tubes' function
def test_collate_tubes(sample_tube_list):
    collated = collate_tubes(sample_tube_list)

    # Check if the collated properties have the correct sizes
    assert collated.a.size() == (len(sample_tube_list), 3)
    assert collated.b.size() == (len(sample_tube_list), 3)
    assert collated.r1.size() == (len(sample_tube_list), 1)
    assert collated.r2.size() == (len(sample_tube_list), 1)
