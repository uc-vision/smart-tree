from pathlib import Path

import click
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from py_structs.torch import map_tensors
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from smart_tree.dataset.dataset import SingleTreeInference, load_dataloader
from smart_tree.model.sparse import batch_collate, sparse_from_batch
from smart_tree.util.file import load_data_npz, load_o3d_cloud
from smart_tree.util.mesh.geometries import o3d_merge_clouds
from smart_tree.util.visualizer.camera import o3d_headless_render
from smart_tree.util.visualizer.view import o3d_cloud, o3d_viewer
from smart_tree.data_types.cloud import Cloud, LabelledCloud


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
    ):
        print("Initalizing Model Inference")
        self.device = device
        self.model = load_model(model_path, weights_path, self.device)
        print("Model Loaded")
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.buffer_size = buffer_size

        self.num_workers = num_workers
        self.batch_size = batch_size

    def forward(self, cloud: Cloud, return_masked=True):
        outputs, inputs, masks = [], [], []

        dataloader = load_dataloader(
            cloud,
            self.voxel_size,
            self.block_size,
            self.buffer_size,
            self.num_workers,
            self.batch_size,
        )

        for features, coordinates, mask in tqdm(
            dataloader, desc="Inferring", leave=False
        ):
            sparse_input = sparse_from_batch(
                features[:, :3],
                coordinates,
                device=self.device,
            )

            out = self.model.forward(sparse_input)

            inputs.append(features.detach().cpu())
            outputs.append(out.detach().cpu())
            masks.append(mask.detach().cpu())

        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        masks = torch.cat(masks)

        vector = torch.exp(outputs[:, [0]]) * outputs[:, 1:4]
        class_l = torch.argmax(outputs[:, 4:], dim=1)

        lc: LabelledCloud = LabelledCloud(
            xyz=inputs[:, :3],
            rgb=inputs[:, 3:6],
            vector=vector,
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


def test():
    voxel_size = 0.001
    block_size = 4
    buffer_size = 0.4

    # data_path = Path("training-data-segmentation/beech_weeping/beech_weeping_1.npz")

    # model_path = Path(
    #     "/smart-tree/Prescient_Tree/model/weights/glowing-monkey-66_model.pt")
    # weights_path = Path(
    #     "/smart-tree/Prescient_Tree/model/weights/glowing-monkey-66_model_weights.pt")

    data_path = "/mnt/ssd/PhD/smart-tree/data/vine/nerf_allign.pcd"

    model_path = Path(
        "/mnt/ssd/PhD/smart-tree/smart_tree/model/weights/vines/proud-plant-157_model.pt"
    )
    weights_path = Path(
        "/mnt/ssd/PhD/smart-tree/smart_tree/model/weights/vines/proud-plant-157_model_weights.pt"
    )

    # cloud, skeleton = load_data_npz(data_path)

    cloud = Cloud.from_o3d_cld(load_o3d_cloud(data_path))

    inferer = ModelInference(
        model_path, weights_path, voxel_size, block_size, buffer_size
    )

    outputs, inputs, masks = inferer.forward(cloud)

    medial_clouds = []
    clouds = []

    for xyz, out, mask in zip(outputs, inputs, masks):
        c = np.random.rand(3)

        mask = mask.reshape(-1)

        xyz = xyz[:, :3]  # [mask]
        radius = out[:, [0]]  # [mask]
        direction = out[:, 1:4]  # [mask]

        new_xyz = xyz + torch.exp(radius) * direction

        clouds.append(o3d_cloud(xyz.reshape(-1, 3), colour=c))
        medial_clouds.append(o3d_cloud(new_xyz.reshape(-1, 3), colour=c))

    o3d_view_geometries([o3d_merge_clouds(clouds), o3d_merge_clouds(medial_clouds)])


@click.command()
@click.option(
    "--scene_config_path",
    default="conf/scene/default.yaml",
    type=str,
    prompt="config path?",
    help="Location of scene yaml config.",
)
def main():
    pass


if __name__ == "__main__":
    test()
    # main()
