from typing import List, Tuple, Union

import torch

from smart_tree.data_types.cloud import Cloud

from .function_timer import timer


def ravel_hash(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, x.shape
    x = x - x.min(dim=0)[0]
    x = x.to(torch.int64)
    xmax = x.max(dim=0)[0].to(torch.int64) + 1
    h = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(
    coords: torch.Tensor,
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
) -> List[torch.Tensor]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = (voxel_size,) * 3
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
    coords = torch.floor(coords / voxel_size).to(torch.int64)
    unique_coords, inverse_indices, counts = torch.unique(
        ravel_hash(coords), return_inverse=True, return_counts=True, dim=0
    )
    indices = torch.cumsum(counts, dim=0) - 1
    coords = coords[indices]
    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

@timer
def voxelize_cloud(cloud: Cloud, voxel_size: float, use_xyz: True, use_rgb: False):
    
    feats = []
    if use_xyz:
        feats.append(cloud.xyz)
    if use_rgb:
        feats.append(cloud.rgb)
    
    feats = torch.cat(feats, 1) if len(feats) > 1 else feats[0]

    coords, feats = cloud.xyz // voxel_size, feats
    coords -= torch.min(coords, axis=0, keepdims=True)[0]
    coords, indices, inverse_indices = sparse_quantize(coords, 0.5, return_index=True, return_inverse=True)
    voxel_coords = coords
    voxel_feats = feats[indices]

    return voxel_feats, voxel_coords, inverse_indices