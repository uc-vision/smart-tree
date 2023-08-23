import torch
from torch import Tensor, rand
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from smart_tree.data_types.branch import BranchSkeleton

patch_typeguard()  # use before @typechecked


def test_branch_skeleton_type_error():
    xyz = torch.rand(100)
    radii = torch.rand(100)

    BranchSkeleton(0, 0, xyz, radii)


if __name__ == "__main__":
    test_branch_skeleton_type_error()
