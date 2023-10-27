from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud, LabelledCloud
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
        radius, medial_direction, branch_direction, class_l = [], [], [], []

        for input_feats, coords, mask, filename in tqdm(
            self.dataloader(cloud), leave=False, desc="Model Inference"
        ):
            sparse_input = sparse_from_batch(
                input_feats,
                coords,
                device=self.device,
            )

            preds = self.model.forward(sparse_input)

            mask = mask.view(-1)

            radius.append(preds["radius"][mask])
            medial_direction.append(preds["medial_direction"][mask])
            #branch_direction.append(preds["branch_direction"][mask])
            class_l.append(preds["class_l"][mask])

            inputs.append(input_feats[mask])

        radius = torch.cat(radius)
        medial_direction = torch.cat(medial_direction)
        class_l = torch.cat(class_l)
        #branch_direction = torch.cat(branch_direction)
        inputs = torch.cat(inputs)

        medial_vector = torch.exp(radius) * medial_direction

        class_l = torch.argmax(class_l, dim=1, keepdim=True)

        return LabelledCloud(
            xyz=inputs[:, :3],
            # rgb=inputs[:, 3:6],
            #branch_direction=branch_direction,
            medial_vector=medial_vector,
            class_l=class_l,
        ).to_device(self.device)
