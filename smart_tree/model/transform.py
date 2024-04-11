from dataclasses import dataclass
import torch
from typing import List
from spconv.pytorch.utils import PointToVoxel

from ..data_types.cloud import Cloud


@dataclass
class VoxelizedTrainingData:
    input_voxelized: torch.Tensor
    coords: torch.Tensor
    targets: torch.Tensor
    voxel_id: torch.Tensor
    filename: str | List[str]


# voxel_gen = PointToVoxel(
#         vsize_xyz=[0.005] * 3,
#         coors_range_xyz=[-4,-4,-4,4,4,4],
#         num_point_features=3,
#         max_num_voxels=2000000,
#         max_num_points_per_voxel=1,
#         device=torch.device("cuda"),
# )


def sparse_voxelize(
    cld: Cloud,
    voxel_size: float,
    use_xyz: bool,
    use_rgb: bool,
    fp_16: bool = False,
):

    features = []

    input_feat_size = 0

    if use_xyz:
        features.append(cld.xyz)
        input_feat_size += 3
    if use_rgb:
        features.append(cld.rgb)
        input_feat_size += 3

    if hasattr(cld, "class_l"):

        features.append(cld.class_l)

    if len(features) > 0:
        features = torch.cat(features, dim=1)
    else:
        features = features[0]

    # print(*cld.min_xyz, *cld.max_xyz, len(cld))

    voxel_gen = PointToVoxel(
        vsize_xyz=[voxel_size] * 3,
        coors_range_xyz=[*cld.min_xyz, *cld.max_xyz],
        num_point_features=features.shape[1],
        max_num_voxels=len(cld),
        max_num_points_per_voxel=1,
        device=cld.device,
    )

    # feats_voxelized, coords, num_pts, voxel_id = voxel_gen.generate_voxel_with_id(
    #     features
    # )

    feats_voxelized, coords, num_pts = voxel_gen(features)

    feats_voxelized = feats_voxelized.squeeze(1)

    coords = coords.squeeze(1)
    indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)

    coords = torch.cat((indice, coords), dim=1)

    dtype = torch.float16 if fp_16 else torch.float32

    input_feats, target_feats = (
        feats_voxelized[:, :input_feat_size].to(dtype),
        feats_voxelized[:, [-1]],
    )

    return VoxelizedTrainingData(
        input_feats,
        coords,
        target_feats,
        input_feats,
        cld.filename,
    )


def merge_voxelized_data(data: List[VoxelizedTrainingData]) -> VoxelizedTrainingData:
    for i, d in enumerate(data, start=0):
        d.coords += i

    input_voxelized = torch.cat([data.input_voxelized for data in data])
    targets = torch.cat([data.targets for data in data])
    coords = torch.cat([data.coords for data in data])
    voxel_id = torch.cat([data.voxel_id for data in data])

    filenames = [data.filename for data in data]

    merged_data = VoxelizedTrainingData(
        input_voxelized=input_voxelized,
        coords=coords,
        targets=targets,
        voxel_id=voxel_id,
        filename=filenames,
    )

    return merged_data


# def sparse_voxelize_clouds(
#     clds: List[Cloud],
#     voxel_size: float,
#     input_feature_names: List[str],
#     target_feature_names: List[str],
# ) -> List[Dict]:
#     """Function to voxelize a list of clouds."""
#     return [
#         sparse_voxelize(cld, voxel_size, input_feature_names, target_feature_names)
#         for cld in clds
#     ]


# def extract_cloud_features(cld, feature_names):
#     return {k: getattr(cld, k) for k in feature_names}


# def get_cloud_feature_dims(features, dim=1):
#     return {k: v.shape[dim] for k, v in features.items()}


# def dict_to_concated_tensor(dict: Dict) -> torch.tensor:
#     return torch.cat([v for v in dict.values()], dim=1)


# def filter_dict(original_dict, keys_to_keep):
#     return {k: original_dict[k] for k in keys_to_keep if k in original_dict}


# def concated_tensor_to_dict(tensor: torch.tensor, feats_shape: Dict):
#     slice_index = 0
#     return_dict = {}

#     for k, v in feats_shape.items():
#         return_dict[k] = tensor[:, slice_index : slice_index + v]
#         slice_index += v
#     return return_dict


# def sparse_voxelize(
#     cld: Cloud,
#     voxel_size: float,
#     input_feature_names: List[str],
#     target_feature_names: List[str] = [],
#     extract_point_ids=True,
#     extract_masks=True,
#     use_number_pts=False,
# ) -> Dict:
#     all_feats_dict = extract_cloud_features(
#         cld,
#         input_feature_names + target_feature_names,
#     )

#     all_feats_dims = get_cloud_feature_dims(all_feats_dict)
#     all_feats_tensor = dict_to_concated_tensor(all_feats_dict)

#     target_feats = extract_cloud_features(cld, target_feature_names)

#     class_mask = (
#         list(extract_cloud_features(cld, ["class_loss_mask"]).values())[0]
#         if extract_masks
#         else None
#     )

#     point_id = (
#         list(extract_cloud_features(cld, ["point_ids"]).values())[0]
#         if extract_point_ids
#         else None
#     )

#     voxel_gen = PointToVoxel(
#         vsize_xyz=[voxel_size] * 3,
#         coors_range_xyz=[*cld.min_xyz, *cld.max_xyz],
#         num_point_features=all_feats_tensor.shape[1],
#         max_num_voxels=len(cld),
#         max_num_points_per_voxel=1,
#         device=cld.device,
#     )

#     all_feats_voxelized, coords, num_pts, voxel_id = voxel_gen.generate_voxel_with_id(
#         all_feats_tensor
#     )

#     all_feats_voxelized = all_feats_voxelized.squeeze(1)
#     coords = coords.squeeze(1)
#     indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
#     coords = torch.cat((indice, coords), dim=1)

#     all_feats_dict: Dict = concated_tensor_to_dict(all_feats_voxelized, all_feats_dims)

#     input_feats = filter_dict(all_feats_dict, input_feature_names)
#     target_feats = filter_dict(all_feats_dict, target_feature_names)

#     if use_number_pts:
#         input_feats["num_pts"] = num_pts

#     return VoxelizedTrainingData(
#         input_feats,
#         coords,
#         target_feats,
#         voxel_id,
#         cld.filename,
#         class_mask,
#         point_id,
#     )
