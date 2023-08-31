import contextlib
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import List

import hydra
import numpy as np
import open3d as o3d
import taichi as ti
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from py_structs.torch import map_tensors
from sklearn.metrics import f1_score, mean_absolute_percentage_error
from tqdm import tqdm

import wandb
from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import TreeDataset
from smart_tree.model.helper import get_batch, model_output_to_labelled_clds
from smart_tree.model.loss import compute_loss
from smart_tree.model.model import Smart_Tree
from smart_tree.model.sparse import batch_collate, sparse_from_batch, split_sparse
from smart_tree.o3d_abstractions.camera import Renderer, o3d_headless_render
from smart_tree.o3d_abstractions.geometries import o3d_cloud
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.util.maths import torch_normalized
from smart_tree.util.misc import concate_dict_of_tensors, flatten_list


def view(loader, cfg):
    for sp_input, targets, mask, filenames in tqdm(
        get_batch(loader, torch.device("cuda"), cfg.fp16)
    ):
        split = split_sparse(sp_input)

        clds = []

        for coords, sp_feats in split:
            # clds.append(Cloud(sp_feats.reshape(-1, 3)).to_o3d_cld())
            # clds.append(Cloud(sp_feats.reshape(-1, 3)).to_o3d_cld())
            print(len(Cloud(sp_feats.reshape(-1, 3))))
        # o3d_viewer(clds)


@hydra.main(
    version_base=None,
    config_path="../conf/vines",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    ti.init(arch=ti.gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader, drop_last=False)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    view(test_loader, cfg)

    view(train_loader, cfg)
    view(val_loader, cfg)


if __name__ == "__main__":
    main()
