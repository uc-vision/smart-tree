from itertools import repeat
from typing import List, Tuple, Union, Optional

from tensordict import TensorDict

import numpy as np
import spconv.pytorch as spconv
import torch

from dataclasses import asdict, dataclass

from .transform import TransformedCloud
from ..data_types.cloud import merge_clouds


@dataclass
class SparseVoxelTensor:
    features: torch.tensor
    indices: torch.tensor
    spatial_shape: torch.tensor
    batch_size: torch.tensor

    def to(self, device: torch.device):
        data = asdict(self)
        data["features"] = self.features.to(device)
        data["indices"] = self.indices.to(device)

        return SparseVoxelTensor(**data)


def merge_voxelized_cloud(data: List[TransformedCloud]) -> TransformedCloud:
    for i, d in enumerate(data, start=0):
        d.voxel_coords += i

    voxel_features = torch.cat([data.voxel_features for data in data])
    voxel_targets = torch.cat([data.voxel_targets for data in data])
    voxel_coords = torch.cat([data.voxel_coords for data in data])

    voxel_ids = None
    if all(d.voxel_ids is not None for d in data):
        voxel_ids = torch.cat([d.voxel_ids for d in data])

    filenames = None
    if all(d.filename is not None for d in data):
        filenames = [d.filename for d in data]

    voxel_mask = None
    if all(d.voxel_mask is not None for d in data):
        voxel_mask = torch.cat([d.voxel_mask for d in data])

    cloud = None
    if all(d.input_cloud is not None for d in data):
        cloud = merge_clouds([d.input_cloud for d in data])

    return TransformedCloud(
        voxel_features,
        voxel_targets,
        voxel_coords,
        voxel_ids=voxel_ids,
        voxel_mask=voxel_mask,
        input_cloud=cloud,
    )


def cloud_to_sparse_tensor(cloud: TransformedCloud):
    values, _ = torch.max(cloud.voxel_coords, 0)
    batch_size = values[0] + 1

    return SparseVoxelTensor(
        cloud.voxel_features,
        cloud.voxel_coords,
        values[1:],
        batch_size,
    )


# def batch_collate(batch: List[TransformedCloud]):

#     transformed = merge_voxelized_cloud(batch)
#     batch_size = len(batch)


#     sparse_tensor = transform_cloud_to_sparse_tensor(transformed, batch_size)

#     return (
#         sparse_tensor,
#         transformed.voxel_ids,
#         transformed.voxel_mask,
#         transformed.input_cloud,
#     )

# def split_sparse(sparse_tensor):
#     cloud_ids = sparse_tensor.indices[:, 0]
#     num_clouds = cloud_ids.max() + 1
#     spatial_shape = sparse_tensor.spatial_shape
#     split_sparse = []
#     for i in range(num_clouds):
#         feats = sparse_tensor.features[cloud_ids == i]
#         coords = sparse_tensor.indices[cloud_ids == i]
#         coords = sparse_tensor.indices[cloud_ids == i]


#     return


    # return [SparseVoxelTensor(feats, coords, spatial_shape, )


    # return [
    #     (sparse_tensor.indices[cloud_ids == i], sparse_tensor.features[cloud_ids == i])
    #     for i in range(num_clouds)
    # ]





# def sparse_from_batch(features, coordinates, device):
#     batch_size = features.shape[0]

#     features = features.to(device)
#     coordinates = coordinates.to(device)

#     values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

#     return spconv.SparseConvTensor(
#         features, coordinates.int(), values[1:], batch_size=batch_size
#     )


# def split_sparse_list(indices, features):
#     cloud_ids = indices[:, 0]

#     num_clouds = cloud_ids.max() + 1
#     return [
#         (indices[cloud_ids == i], features[cloud_ids == i]) for i in range(num_clouds)
#     ]


# def split_sparse(sparse_tensor):
#     cloud_ids = sparse_tensor.indices[:, 0]
#     num_clouds = cloud_ids.max() + 1
#     return [
#         (sparse_tensor.indices[cloud_ids == i], sparse_tensor.features[cloud_ids == i])
#         for i in range(num_clouds)
#     ]


# def merge_voxelized_cloud(data: List[VoxelizedCloud]) -> VoxelizedCloud:
#     for i, d in enumerate(data, start=0):
#         d.voxel_coordinates += i

#     input_voxel_features = torch.cat([data.input_voxel_features for data in data])
#     target_voxel_features = torch.cat([data.target_voxel_features for data in data])
#     voxel_coordinates = torch.cat([data.voxel_coordinates for data in data])
#     voxel_ids = torch.cat([data.voxel_ids for data in data])

#     filenames = [data.filename for data in data]

#     return VoxelizedCloud(
#         input_voxel_features,
#         target_voxel_features,
#         voxel_coordinates,
#         voxel_ids,
#         filenames,
#     )


# def voxel_cloud_to_sparse_tensor(cloud: VoxelizedCloud, batch_size: int):
#     values, _ = torch.max(cloud.voxel_coordinates, 0)

#     return SparseVoxelTensor(
#         cloud.input_voxel_features,
#         cloud.voxel_coordinates,
#         values[1:],
#         batch_size,
#     )


# cloud: VoxelizedCloud = merge_voxelized_cloud(batch)

# sparse_input = voxel_cloud_to_sparse_tensor(cloud, len(batch))

# return sparse_input, cloud.target_voxel_features


# def batch_collate(batch: List[VoxelizedCloud]):

#     pri


#     """Custom Batch Collate Function for Sparse Data..."""

#     block_clouds, batch_feats, batch_coords, batch_mask, voxel_id, fn = zip(*batch)

#     for i, coords in enumerate(batch_coords):
#         coords[:, 0] = torch.tensor([i], dtype=torch.float32)

#     if isinstance(batch_feats[0], tuple):
#         input_feats, target_feats = tuple(zip(*batch_feats))

#         input_feats, target_feats, coords, mask, voxel_id = [
#             torch.cat(x)
#             for x in [input_feats, target_feats, batch_coords, batch_mask, voxel_id]
#         ]

#         return [(input_feats, target_feats), coords, mask, voxel_id, fn]

#     feats, coords, mask, voxel_id = [
#         torch.cat(x) for x in [batch_feats, batch_coords, batch_mask, voxel_id]
#     ]

#     return [feats, coords, mask, voxel_id, fn]


# def batch_collate(batch):
#     """Custom Batch Collate Function for Sparse Data..."""

#     block_clouds, batch_feats, batch_coords, batch_mask, voxel_id, fn = zip(*batch)

#     for i, coords in enumerate(batch_coords):
#         coords[:, 0] = torch.tensor([i], dtype=torch.float32)

#     if isinstance(batch_feats[0], tuple):
#         input_feats, target_feats = tuple(zip(*batch_feats))

#         input_feats, target_feats, coords, mask, voxel_id = [
#             torch.cat(x)
#             for x in [input_feats, target_feats, batch_coords, batch_mask, voxel_id]
#         ]

#         return [(input_feats, target_feats), coords, mask, voxel_id, fn]

#     feats, coords, mask, voxel_id = [
#         torch.cat(x) for x in [batch_feats, batch_coords, batch_mask, voxel_id]
#     ]

#     return [feats, coords, mask, voxel_id, fn]


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(
    coords,
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs
