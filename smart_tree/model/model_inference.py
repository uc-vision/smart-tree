from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from spconv.pytorch.utils import gather_features_by_pc_voxel_id
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud, LabelledCloud  # , merge_clouds
from smart_tree.dataset.dataset import load_dataloader
from torch.nn import Module

from .sparse import cloud_to_sparse_tensor
from .transform import sparse_voxelize
from torch.utils.data import DataLoader
from smart_tree.dataset.dataset import SingleTreeInference

from spconv.pytorch.utils import PointToVoxel
from .voxelize import SparseVoxelizer

""" Loads model and model weights, then returns the input, outputs and mask """


class ModelInference:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,  # partial
        dataset: SingleTreeInference,  # partial
        device=torch.device("cuda:0"),
    ):
        self.model = model.to(device)
        self.data_loader = dataloader
        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def forward(self, cloud: Cloud | LabelledCloud):

        cloud = cloud.to_device(self.device)
        data_loader = self.data_loader(self.dataset(cloud))

        clouds: List[Cloud]
        for clouds in data_loader:

            print(clouds)

            Cloud(clouds[0].voxel_features[:, :3]).view()

            # voxelized_clouds = self.voxelizer.voxelize_clouds(clouds)

            # Voxelize clouds?

            pass

        quit()

        # voxel_features, voxel_coordinates, num_pts, voxel_ids = (
        #    voxel_gen.generate_voxel_with_id(cloud.xyz)
        # )

        #     pt_radius = gather_features_by_pc_voxel_id(preds["radius"] , voxel_cloud.voxel_ids)
        #     pt_direction = gather_features_by_pc_voxel_id( preds["direction"], voxel_cloud.voxel_ids)
        #     pt_class = gather_features_by_pc_voxel_id(preds["class_l"], voxel_cloud.voxel_ids)
        #     pt_mask = gather_features_by_pc_voxel_id(voxel_cloud.voxel_mask, voxel_cloud.voxel_ids).bool()

        # quit()

        # clouds = []

        # dataloader = load_dataloader(
        #     cloud,
        #     self.voxel_size,
        #     self.block_size,
        #     self.buffer_size,
        #     self.num_workers,
        #     self.batch_size,
        # )

        # for cloud in tqdm(dataloader, desc="Inferring", leave=False):

        #     voxel_cloud = sparse_voxelize(
        #         cloud,
        #         self.voxel_size,
        #         use_xyz=True,
        #         use_rgb=False,
        #         voxelize_radius=False,
        #         voxelize_direction=False,
        #         voxelize_class=False,
        #         voxelize_mask=True,
        #     )

        #     sparse_input = cloud_to_sparse_tensor(voxel_cloud)
        #     sparse_input = sparse_input.to(self.device)

        #     preds = self.model.forward(sparse_input)

        #     pt_radius = gather_features_by_pc_voxel_id(preds["radius"] , voxel_cloud.voxel_ids)
        #     pt_direction = gather_features_by_pc_voxel_id( preds["direction"], voxel_cloud.voxel_ids)
        #     pt_class = gather_features_by_pc_voxel_id(preds["class_l"], voxel_cloud.voxel_ids)
        #     pt_mask = gather_features_by_pc_voxel_id(voxel_cloud.voxel_mask, voxel_cloud.voxel_ids).bool()

        #     class_l = torch.argmax(pt_class, dim=1, keepdim=True)
        #     medial_vector = torch.exp(pt_radius) * pt_direction

        #     cld = Cloud(
        #         xyz=cloud.xyz.cpu(),
        #         medial_vector=medial_vector.cpu(),
        #         class_l=class_l,
        #     )        self.device = device
        # self.verbose = verbose
        # self.voxel_size = voxel_size
        # self.block_size = block_size
        # self.buffer_size = buffer_size

        # self.num_workers = num_workers
        # self.batch_size = batch_size

        # self.model = model.to(device)

        #     cld = cld.filter(pt_mask)

        #     clouds.append(cld)

        # #return merge_clouds(clouds)

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
