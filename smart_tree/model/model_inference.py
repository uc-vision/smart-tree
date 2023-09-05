
from typing import Optional

import torch
from torch.utils.data import DataLoader

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import SingleTreeInference
from smart_tree.model.sparse.util import sparse_from_batch


def load_model(model_path, weights_path, device=torch.device("cuda:0")):
    model = torch.load(f"{model_path}", map_location=device)
    model.load_state_dict(torch.load(f"{weights_path}"))
    model.eval()
    return model


def load_dataloader(
    cloud: Cloud,
    block_size: float,
    buffer_size: float,
    num_workers: int,
    batch_size: int,
    collate_fn: callable,
    transform: Optional[callable] = None,
    augmentation: Optional[callable] = None,
):
    dataset = SingleTreeInference(
        cloud,
        block_size,
        buffer_size,
        augmentation,
        transform,
    )
    return DataLoader(dataset, batch_size, num_workers, collate_fn=collate_fn)


class ModelInference:
    def __init__(
        self,
        model,
        dataloader,
        device=torch.device("cuda:0"),
    ):
        self.model = model
        self.data_loader = data_loader

    @torch.no_grad()
    def forward(self, cloud: Cloud):
        dataloader = load_dataloader(cloud)

        print("YEYEYYE")

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
