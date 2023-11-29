from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


@dataclass
class VoxelizedTrainingData:
    input_voxelized: Dict[str, torch.Tensor]
    coords: torch.Tensor
    targets: Dict[str, torch.Tensor]
    point_id: torch.Tensor
    voxel_id: torch.Tensor
    mask: torch.Tensor


def merge_voxelized_data(
    data_list: List[VoxelizedTrainingData],
) -> VoxelizedTrainingData:
    merged_input_voxelized = {}
    merged_targets = {}

    for i, data in enumerate(data_list):
        data.coords[:, 0] = torch.tensor([i], dtype=torch.float32)
        data.point_id[:, 0] = torch.tensor([i], dtype=torch.float32)

    merged_coords = torch.cat([data.coords for data in data_list], dim=0)
    merged_point_id = torch.cat([data.point_id for data in data_list], dim=0)

    merged_voxel_id = torch.cat([data.voxel_id for data in data_list], dim=0)
    merged_mask = torch.cat([data.mask for data in data_list], dim=0)

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

    return VoxelizedTrainingData(
        input_voxelized=merged_input_voxelized,
        coords=merged_coords,
        targets=merged_targets,
        point_id=merged_point_id,
        voxel_id=merged_voxel_id,
        mask=merged_mask,
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


def get_cloud_feature_shapes(features):
    return {k: v.shape[1] for k, v in features.items()}


def dict_to_concated_tensor(dicts: List[Dict]) -> torch.tensor:
    return torch.cat([value for d in dicts for value in d.values()], dim=1)


def concated_tensor_to_dict(tensor: torch.tensor, feats_shape: Dict):
    slice_index = 0
    return_dict = {}

    for k, v in feats_shape.items():
        return_dict[k] = tensor[:, slice_index : slice_index + v]
        slice_index += v
    return return_dict


def sparse_voxelize(
    cld: Union[Cloud, LabelledCloud],
    voxel_size: float,
    input_feature_names: List[str],
    target_feature_names: List[str],
) -> Dict:
    input_feats = extract_cloud_features(cld, input_feature_names)
    target_feats = extract_cloud_features(cld, target_feature_names)
    mask = list(extract_cloud_features(cld, ["loss_mask"]).values())[0]
    point_id = list(extract_cloud_features(cld, ["point_ids"]).values())[0]

    feats_shape = get_cloud_feature_shapes(input_feats)
    input_feats = dict_to_concated_tensor([input_feats])

    voxel_gen = PointToVoxel(
        vsize_xyz=[voxel_size] * 3,
        coors_range_xyz=[*cld.min_xyz, *cld.max_xyz],
        num_point_features=input_feats.shape[1],
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    input_feats, coords, num_pts, voxel_id = voxel_gen.generate_voxel_with_id(
        input_feats
    )

    input_feats = input_feats.squeeze(1)
    coords = coords.squeeze(1)
    indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    coords = torch.cat((indice, coords), dim=1)

    input_feats: Dict = concated_tensor_to_dict(input_feats, feats_shape)

    return VoxelizedTrainingData(
        input_feats,
        coords,
        target_feats,
        point_id,
        voxel_id,
        mask,
    )
