from dataclasses import dataclass
from typing import Dict, List, Tuple

import spconv.pytorch as spconv
import torch

from .transform import VoxelizedTrainingData, merge_voxelized_data


@dataclass
class Data:
    point_id: torch.Tensor
    voxel_id: torch.Tensor
    mask: torch.Tensor
    filename: str


def sparse_from_batch(features, coordinates, batch_size, device):
    features = features.to(device)
    coordinates = coordinates.to(device)

    values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

    return spconv.SparseConvTensor(
        features,
        coordinates.int(),
        values[1:],
        batch_size,
    )


def batch_collate(
    batch: List[VoxelizedTrainingData],
    device=torch.device("cuda"),
    fp_16=True,
):
    # dtype = torch.float16 if fp_16 else torch.float32

    training_data: VoxelizedTrainingData = merge_voxelized_data(batch)

    sparse_input = torch.cat(list(training_data.input_voxelized.values()), dim=0)

    sparse_input = sparse_from_batch(
        sparse_input,
        training_data.coords,
        len(batch),
        device,
    )

    return (
        sparse_input,
        training_data.targets,
        Data(
            training_data.point_id,
            training_data.voxel_id,
            training_data.mask,
            training_data.filename,
        ),
    )


def batch_collate_dual_clouds(
    batch: List[Tuple[Dict, Dict]],
    device=torch.device("cuda"),
    fp_16=True,
):
    # Need to batch clouds for each row corresponds to the different augmentation

    batch_1, batch_2 = zip(*batch)

    sparse_input_1, targets_1, data_1 = batch_collate(batch_1, device, fp_16)

    sparse_input_2, targets_2, data_2 = batch_collate(batch_2, device, fp_16)

    return (sparse_input_1, sparse_input_2), (targets_1, targets_2), (data_1, data_2)
