from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


@dataclass
class VoxelizedTrainingData:
    input_voxelized: Dict[str, torch.Tensor]
    coords: torch.Tensor
    targets: Dict[str, torch.Tensor]
    voxel_id: torch.Tensor
    filename: str | List[str]
    mask: Optional[torch.Tensor] = None
    point_id: Optional[torch.Tensor] = None


def merge_voxelized_data(
    data_list: List[VoxelizedTrainingData],
) -> VoxelizedTrainingData:
    merged_input_voxelized = {}
    merged_targets = {}

    # Initialize lists to store point_ids and masks if they exist
    point_ids = []
    masks = []
    file_names = []

    for i, data in enumerate(data_list):
        data.coords[:, 0] = torch.tensor([i], dtype=torch.float32)

        if data.point_id is not None:
            data.point_id[:, 0] = torch.tensor([i], dtype=torch.float32)
            point_ids.append(data.point_id)

        if data.mask is not None:
            masks.append(data.mask)

    merged_coords = torch.cat([data.coords for data in data_list], dim=0)
    merged_voxel_id = torch.cat([data.voxel_id for data in data_list], dim=0)

    # Merge point_ids and masks if they are not empty
    merged_point_id = torch.cat(point_ids, dim=0) if point_ids else None
    merged_mask = torch.cat(masks, dim=0) if masks else None

    for data in data_list:
        for key, tensor in data.input_voxelized.items():
            if key in merged_input_voxelized:
                merged_input_voxelized[key] = torch.cat(
                    (merged_input_voxelized[key], tensor), dim=0
                )
            else:
                merged_input_voxelized[key] = tensor.clone()

        for key, tensor in data.targets.items():
            if key in merged_targets:
                merged_targets[key] = torch.cat((merged_targets[key], tensor), dim=0)
            else:
                merged_targets[key] = tensor.clone()

        file_names.append(data.filename)

    return VoxelizedTrainingData(
        input_voxelized=merged_input_voxelized,
        coords=merged_coords,
        targets=merged_targets,
        point_id=merged_point_id,
        voxel_id=merged_voxel_id,
        mask=merged_mask,
        filename=file_names,  # Assuming all data have the same filename
    )


def sparse_voxelize_clouds(
    clds: List[Union[Cloud, LabelledCloud]],
    voxel_size: float,
    input_feature_names: List[str],
    target_feature_names: List[str],
) -> List[Dict]:
    """Function to voxelize a list of clouds."""
    return [
        sparse_voxelize(cld, voxel_size, input_feature_names, target_feature_names)
        for cld in clds
    ]


def extract_cloud_features(cld, feature_names):
    return {k: getattr(cld, k) for k in feature_names}


def get_cloud_feature_dims(features, dim=1):
    return {k: v.shape[dim] for k, v in features.items()}


def dict_to_concated_tensor(dict: Dict) -> torch.tensor:
    return torch.cat([v for v in dict.values()], dim=1)


def filter_dict(original_dict, keys_to_keep):
    return {k: original_dict[k] for k in keys_to_keep if k in original_dict}


def concated_tensor_to_dict(tensor: torch.tensor, feats_shape: Dict):
    slice_index = 0
    return_dict = {}

    for k, v in feats_shape.items():
        return_dict[k] = tensor[:, slice_index : slice_index + v]
        slice_index += v
    return return_dict


def sparse_voxelize(
    cld: Cloud | LabelledCloud,
    voxel_size: float,
    input_feature_names: List[str],
    target_feature_names: List[str] = [],
    extract_point_ids=True,
    extract_masks=True,
) -> Dict:
    all_feats_dict = extract_cloud_features(
        cld, input_feature_names + target_feature_names
    )
    all_feats_dims = get_cloud_feature_dims(all_feats_dict)
    all_feats_tensor = dict_to_concated_tensor(all_feats_dict)

    target_feats = extract_cloud_features(cld, target_feature_names)

    if extract_masks:
        mask = list(extract_cloud_features(cld, ["loss_mask"]).values())[0]
    else:
        mask = None

    if extract_point_ids:
        point_id = list(extract_cloud_features(cld, ["point_ids"]).values())[0]
    else:
        point_id = None

    voxel_gen = PointToVoxel(
        vsize_xyz=[voxel_size] * 3,
        coors_range_xyz=[*cld.min_xyz, *cld.max_xyz],
        num_point_features=all_feats_tensor.shape[1],
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    all_feats_voxelized, coords, num_pts, voxel_id = voxel_gen.generate_voxel_with_id(
        all_feats_tensor
    )

    all_feats_voxelized = all_feats_voxelized.squeeze(1)
    coords = coords.squeeze(1)
    indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    coords = torch.cat((indice, coords), dim=1)

    all_feats_dict: Dict = concated_tensor_to_dict(all_feats_voxelized, all_feats_dims)

    input_feats = filter_dict(all_feats_dict, input_feature_names)
    target_feats = filter_dict(all_feats_dict, target_feature_names)

    return VoxelizedTrainingData(
        input_feats,
        coords,
        target_feats,
        voxel_id,
        cld.filename,
        mask,
        point_id,
    )
