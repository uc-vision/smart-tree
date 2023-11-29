import open3d as o3d
import pytest
import torch

from smart_tree.data_types.branch import BranchSkeleton
from smart_tree.data_types.tube import Tube


@pytest.fixture
def sample_branch_skeleton():
    return BranchSkeleton(
        _id=1,
        parent_id=0,
        xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        radii=torch.tensor([[0.1], [0.2]]),
    )


def test_to_tubes(sample_branch_skeleton):
    tubes = sample_branch_skeleton.as_tubes()

    assert all(isinstance(tube, Tube) for tube in tubes)

    assert len(tubes) == len(sample_branch_skeleton) - 1


def test_length(sample_branch_skeleton):
    length = sample_branch_skeleton.length

    expected_length = torch.norm(
        sample_branch_skeleton.xyz[1:] - sample_branch_skeleton.xyz[:-1],
        dim=1,
    ).sum()

    assert torch.allclose(length, expected_length)


def test_filter(sample_branch_skeleton):
    mask = torch.tensor([True, False])

    print(sample_branch_skeleton)

    filtered_branch_skeleton = sample_branch_skeleton.filter(mask)

    assert isinstance(filtered_branch_skeleton, BranchSkeleton)

    assert torch.equal(filtered_branch_skeleton.xyz, torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.equal(filtered_branch_skeleton.radii, torch.tensor([[0.1]]))
    assert filtered_branch_skeleton._id == sample_branch_skeleton._id
    assert filtered_branch_skeleton.parent_id == sample_branch_skeleton.parent_id
    assert filtered_branch_skeleton.child_id == sample_branch_skeleton.child_id


def test_to_o3d_lineset(sample_branch_skeleton):
    lineset = sample_branch_skeleton.as_o3d_lineset()

    assert isinstance(lineset, o3d.geometry.LineSet)


def test_to_o3d_tube(sample_branch_skeleton):
    tube_mesh = sample_branch_skeleton.as_o3d_tube()

    assert isinstance(tube_mesh, o3d.geometry.TriangleMesh)
