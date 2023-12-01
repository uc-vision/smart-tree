from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

import wandb


def log_cloud_on_wandb(clouds: List, epoch):
    upload_data = {}

    for cloud in tqdm(clouds, desc="Uploading Clouds", leave=False):
        cloud = cloud.to_device(torch.device("cpu"))

        xyzrgb = np.concatenate((cloud.xyz.numpy(), cloud.rgb.numpy() * 255), axis=-1)

        upload_data[f"{Path(cloud.filename).stem}"] = wandb.Object3D(xyzrgb)

    wandb.log(
        upload_data,
        step=epoch,
    )
