from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.dataset.dataset import SingleTreeInference


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
        block_size=block_size,
        buffer_size=buffer_size,
        augmentation=augmentation,
        transform=transform,
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
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def forward(self, cloud: Cloud):
        inputs, masks = [], []
        radius, direction, branch_direction, class_l = [], [], [], []

        for model_input, targets in tqdm(
            self.dataloader(cloud),
            leave=False,
            desc="Model Inference",
        ):

            preds = self.model.forward(model_input)

            radius.append(preds["radius"])
            direction.append(preds["direction"])
            class_l.append(preds["class_l"])

            inputs.append(model_input.features)

        radius = torch.cat(radius)
        direction = torch.cat(direction)

        medial_vector = torch.exp(radius) * direction

        class_l = torch.argmax(torch.cat(class_l), dim=1, keepdim=True)

        return LabelledCloud(
            xyz=torch.cat(inputs)[:, :3],
            # rgb=inputs[:, 3:6],
            # branch_direction=branch_direction,
            medial_vector=medial_vector,
            class_l=class_l,
        ).to_device(self.device)
