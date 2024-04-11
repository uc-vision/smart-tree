from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import spconv.pytorch as spconv
import torch

from .transform import VoxelizedTrainingData, merge_voxelized_data
from ..data_types.cloud import Cloud


@dataclass
class Data:
    point_id: torch.Tensor
    voxel_id: torch.Tensor
    class_mask: torch.Tensor
    vector_mask: torch.Tensor
    filename: str


@dataclass
class SparseVoxelData:
    features: torch.tensor
    indices: torch.tensor
    spatial_shape: torch.tensor
    batch_size: torch.tensor

    def to(self, device: torch.device):
        data = asdict(self)
        data["features"] = self.features.to(device)
        data["indices"] = self.indices.to(device)

        return SparseVoxelData(**data)


def sparse_from_batch(features, coordinates, batch_size):

    values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

    return SparseVoxelData(
        features,
        coordinates.int(),
        values[1:],
        batch_size,
    )

    # return {"features", features, "coordinates": coordinates.int(), "values" : values[1:], "batch_size": batch_size}


def batch_collate(
    batch: List[VoxelizedTrainingData],
):

    training_data: VoxelizedTrainingData = merge_voxelized_data(batch)

    sparse_input = training_data.input_voxelized

    sparse_input = sparse_from_batch(
        sparse_input,
        training_data.coords,
        len(batch),
    )
    return sparse_input, training_data.targets


def split_sparse(sparse_tensor):
    cloud_ids = sparse_tensor.indices[:, 0]
    num_clouds = cloud_ids.max() + 1
    return [
        (sparse_tensor.indices[cloud_ids == i], sparse_tensor.features[cloud_ids == i])
        for i in range(num_clouds)
    ]


def sparse_to_clouds(
    sparse_tensor: spconv.SparseConvTensor,
    predictions,
) -> List[Cloud]:

    class_prediction = torch.argmax(predictions["class_predictions"], 1).reshape(-1, 1)

    increment = 0
    clouds = []
    for coord, feat in split_sparse(sparse_tensor):
        cloud = Cloud(
            xyz=feat,
            class_l=class_prediction[increment : increment + feat.shape[0]],
        ).to_device(torch.device("cpu"))

        increment += feat.shape[0]

        clouds.append(cloud)

    return clouds


def merge_sparse_tensors(
    sparse_tensors: List[spconv.SparseConvTensor],
) -> spconv.SparseConvTensor:
    """
    Merges a list of SparseConvTensor objects into a single SparseConvTensor,
    ensuring indices are of type torch.int32 as required by spconv.

    Args:
        sparse_tensors (List[spconv.SparseConvTensor]): The list of SparseConvTensor objects to merge.

    Returns:
        spconv.SparseConvTensor: The merged SparseConvTensor.
    """
    all_features = []
    all_indices = []
    max_batch_idx = 0

    device = sparse_tensors[0].features.device

    for tensor in sparse_tensors:
        indices = tensor.indices.clone()
        indices[:, 0] += max_batch_idx  # Increment batch indices
        max_batch_idx = indices[:, 0].max().item() + 1

        all_features.append(tensor.features)
        all_indices.append(indices)

    merged_features = torch.cat(all_features, dim=0)
    merged_indices = torch.cat(all_indices, dim=0).to(
        torch.int32
    )  # Convert indices to torch.int32

    spatial_size = (
        merged_indices.max(dim=0).values[1:] + 1
    )  # Calculate spatial size, excluding batch index

    merged_tensor = spconv.SparseConvTensor(
        features=merged_features,
        indices=merged_indices,
        spatial_shape=spatial_size.cpu().numpy(),  # Convert to numpy array
        batch_size=max_batch_idx,
    )

    return merged_tensor
