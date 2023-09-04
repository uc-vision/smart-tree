import torch


def batch_collate(batch):
    """Custom Batch Collate Function for Sparse Data..."""

    batch_feats, batch_coords, batch_mask, fn = zip(*batch)

    for i, coords in enumerate(batch_coords):
        coords[:, 0] = torch.tensor([i], dtype=torch.float32)

    if isinstance(batch_feats[0], tuple):
        input_feats, target_feats = zip(*batch_feats)
        input_feats, target_feats, coords, mask = map(
            torch.cat, [input_feats, target_feats, batch_coords, batch_mask]
        )
        return [(input_feats, target_feats), coords, mask, fn]
    else:
        feats, coords, mask = map(torch.cat, [batch_feats, batch_coords, batch_mask])
        return [feats, coords, mask, fn]
