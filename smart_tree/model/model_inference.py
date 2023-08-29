from pathlib import Path

import click
import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from py_structs.torch import map_tensors
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import SingleTreeInference, load_dataloader
from smart_tree.model.sparse import batch_collate, sparse_from_batch
from smart_tree.util.file import load_data_npz, load_o3d_cloud
from smart_tree.o3d_abstractions.geometries import o3d_merge_clouds, o3d_cloud
from smart_tree.o3d_abstractions.camera import o3d_headless_render
from smart_tree.o3d_abstractions.visualizer import o3d_viewer


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

    def forward(self, cloud: Cloud, return_masked=True):
        inputs, masks = [], []
        radius, medial_direction, branch_direction, class_l = [], [], [], []

        dataloader = load_dataloader(
            cloud,
            self.voxel_size,
            self.block_size,
            self.buffer_size,
            self.num_workers,
            self.batch_size,
        )

        for features, coordinates, mask, filename in tqdm(
            dataloader, desc="Inferring", leave=False
        ):
            sparse_input = sparse_from_batch(
                features[:, :3],
                coordinates,
                device=self.device,
            )

            preds = self.model.forward(sparse_input)

            radius.append(preds["radius"].detach().cpu())

            medial_direction.append(preds["medial_direction"].detach().cpu())

            class_l.append(preds["class_l"].detach().cpu())

            inputs.append(features.detach().cpu())
            masks.append(mask.detach().cpu())

        radius = torch.cat(radius)
        medial_direction = torch.cat(medial_direction)
        class_l = torch.cat(class_l)

        inputs = torch.cat(inputs)
        masks = torch.cat(masks)

        medial_vector = torch.exp(radius) * medial_direction
        class_l = torch.argmax(class_l, dim=1, keepdim=True)

        lc = Cloud(
            xyz=inputs[:, :3],
            rgb=inputs[:, 3:6],
            medial_vector=medial_vector,
            class_l=class_l,
        )

        if return_masked:
            return lc.filter(masks)

        return lc

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
