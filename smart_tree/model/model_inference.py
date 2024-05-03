from pathlib import Path

import torch
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud, merge_clouds
from smart_tree.dataset.dataset import load_dataloader
from spconv.pytorch.utils import gather_features_by_pc_voxel_id
from dataclasses import asdict


def load_model(model_path, weights_path, device=torch.device("cuda:0")):
    model = torch.load(f"{model_path}", map_location=device)
    model.load_state_dict(torch.load(f"{weights_path}"))
    model.eval()

    return model


""" Loads model and model weights, then returns the input, outputs and mask """


class ModelInference:
    def __init__(
        self,
        model_path: Path,
        weights_path: Path,
        voxel_size: float,
        block_size: float,
        buffer_size: float,
        num_workers=8,
        batch_size=4,
        device=torch.device("cuda:0"),
        verbose=False,
    ):
        self.device = device
        self.verbose = verbose
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.buffer_size = buffer_size

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.model = load_model(model_path, weights_path, self.device)

        if self.verbose:
            print("Model Loaded Succesfully")

    @torch.no_grad()
    def forward(self, cloud: Cloud, return_masked=True):

        clouds = []

        dataloader = load_dataloader(
            cloud,
            self.voxel_size,
            self.block_size,
            self.buffer_size,
            self.num_workers,
            self.batch_size,
        )

        for sparse_input, voxel_ids, voxel_mask, cloud in tqdm(
            dataloader, desc="Inferring", leave=False
        ):

            sparse_input = sparse_input.to(self.device)

            preds = self.model.forward(sparse_input)

            voxel_radius = preds["radius"]  # [mask]
            voxel_direction = preds["direction"]  # [mask]
            voxel_class = preds["class_l"]  # [mask]

            # voxel_mask = voxel_mask.bool()

            # print(voxel_ids.max(), voxel_ids.shape)

            pt_radius = gather_features_by_pc_voxel_id(voxel_radius, voxel_ids)
            pt_direction = gather_features_by_pc_voxel_id(voxel_direction, voxel_ids)
            pt_class = gather_features_by_pc_voxel_id(voxel_class, voxel_ids)
            pt_mask = gather_features_by_pc_voxel_id(voxel_mask, voxel_ids).bool()

            # pt_radius = voxel_radius
            # pt_direction = voxel_direction
            # pt_class = voxel_class

            # print(pt_class)

            class_l = torch.argmax(pt_class, dim=1, keepdim=True)
            medial_vector = torch.exp(pt_radius) * pt_direction

            cld = Cloud(
                xyz=cloud.xyz.cpu(),
                medial_vector=medial_vector.cpu(),
                class_l=class_l,
            )
            cld = cld.filter(pt_mask)

            clouds.append(cld)

        merged_cloud = merge_clouds(clouds)
        merged_cloud.view()

        return merge_clouds(clouds)

    @staticmethod
    def from_cfg(cfg):
        return ModelInference(
            model_path=cfg.model_path,
            weights_path=cfg.weights_path,
            voxel_size=cfg.voxel_size,
            block_size=cfg.block_size,
            buffer_size=cfg.buffer_size,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
        )
