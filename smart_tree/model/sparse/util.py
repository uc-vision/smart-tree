from functools import partial
from typing import List, Union

import spconv.pytorch as spconv
import torch
from py_structs.torch import map_tensors
from spconv.pytorch.utils import PointToVoxel

from smart_tree.data_types.cloud import Cloud, LabelledCloud


def batch_collate(batch):
    """Custom Batch Collate Function for Sparse Data..."""

    batch_feats, batch_coords, batch_mask, fn = zip(*batch)

    for i, coords in enumerate(batch_coords):
        coords[:, 0] = torch.tensor([i], dtype=torch.float32)

    if isinstance(batch_feats[0], tuple):
        input_feats, target_feats = tuple(zip(*batch_feats))

        input_feats, target_feats, coords, mask = [
            torch.cat(x) for x in [input_feats, target_feats, batch_coords, batch_mask]
        ]

        return [(input_feats, target_feats), coords, mask, fn]

    feats, coords, mask = [
        torch.cat(x) for x in [batch_feats, batch_coords, batch_mask]
    ]

    return [feats, coords, mask, fn]


def sparse_from_batch(features, coordinates, device):
    batch_size = features.shape[0]

    features = features.to(device)
    coordinates = coordinates.to(device)

    values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

    return spconv.SparseConvTensor(
        features,
        coordinates.int(),
        values[1:],
        batch_size=batch_size,
    )


def get_batch(dataloader, device, fp_16=False):
    for (feats, target_feats), coords, mask, filenames in dataloader:
        if fp_16:
            feats = feats.half()
            target_feats = target_feats.half()
            coords = coords.half()

        sparse_input = sparse_from_batch(
            feats,
            coords,
            device=device,
        )
        targets = map_tensors(
            target_feats,
            partial(
                torch.Tensor.to,
                device=device,
            ),
        )

        yield sparse_input, targets, mask, filenames

    # indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=cld.device)
    # coords = torch.cat((indice, coords), dim=1)
    # feats = feats.squeeze(1)
    # coords = coords.squeeze(1)

    # mask = torch.ones((feats.shape[0], 1), dtype=torch.int32)

    # if target_feats is None:
    #     return [feats, coords, mask, cld.filename]

    # else:
    #     return [
    #         (feats[:, : input_feats.shape[1]], feats[:, input_feats.shape[1] :]),
    #         coords,
    #         mask,
    #         cld.filename,
    #     ]
