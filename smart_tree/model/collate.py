from typing import List

from ..data_types.cloud import Cloud, LabelledCloud
from .voxelize import VoxelizedCloud


def inference_collate(data):
    return data

    # clouds, sparse_data = zip(*data)

    # if len(sparse_data) == 1:
    #     (feats, coords, inverse_indices) = sparse_data
    # else:
    #     (feats, coords, inverse_indices) = zip(*sparse_data)

    # print(feats)

    # return clouds


def voxelized_clouds_to_sparse_tensor(clouds: List[VoxelizedCloud]):

    return False


def split_sparse_tensor(sparse_tensor):

    return
