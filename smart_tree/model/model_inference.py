from pathlib import Path

import torch
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import load_dataloader, SingleTreeInference
from smart_tree.model.sparse.util import sparse_from_batch


class ModelInference:
    def __init__(
        self,
        model,
        data_loader,
        device=torch.device("cuda:0"),
    ):
        self.model = model
        self.data_loader = data_loader

    @torch.no_grad()
    def run(self, path):
        inputs, masks = [], []
        radius, medial_direction, branch_direction, class_l = [], [], [], []

        for input_feats, coords, mask, filename in self.data_loader:
            sparse_input = sparse_from_batch(
                input_feats[:, :3],
                coords,
                device=self.device,
            )

            preds = self.model.forward(sparse_input)

            radius.append(preds["radius"].cpu())
            medial_direction.append(preds["medial_direction"]).cpu()
            class_l.append(preds["class_l"].cpu())

            inputs.append(features.cpu())
            masks.append(mask.cpu())

        radius = torch.cat(radius)
        medial_direction = torch.cat(medial_direction)
        class_l = torch.cat(class_l)

        inputs = torch.cat(inputs)
        mask = torch.cat(masks)

        medial_vector = torch.exp(radius) * medial_direction
        class_l = torch.argmax(class_l, dim=1, keepdim=True)

        lc = LabelledCloud(
            xyz=inputs[:, :3],
            rgb=inputs[:, 3:6],
            medial_vector=medial_vector,
            class_l=class_l,
        ).filter(mask)

        return lc
