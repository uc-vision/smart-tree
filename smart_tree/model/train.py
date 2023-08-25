import contextlib
import logging
import os
from functools import partial

import hydra
import numpy as np
import open3d as o3d
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from py_structs.torch import map_tensors
from sklearn.metrics import f1_score, mean_absolute_percentage_error
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.dataset.dataset import TreeDataset
from smart_tree.model.model import Smart_Tree
from smart_tree.model.sparse import batch_collate, sparse_from_batch
from smart_tree.util.maths import torch_normalized
from smart_tree.o3d_abstractions.geometries import o3d_cloud
from smart_tree.util.misc import concate_dict_of_tensors, flatten_list
from smart_tree.o3d_abstractions.camera import Renderer, o3d_headless_render
from smart_tree.o3d_abstractions.visualizer import o3d_viewer

from .helper import (
    get_batch,
    model_output_to_labelled_clds,
)
from .tracker import Tracker


def train_epoch(
    train_loader,
    model,
    optimizer,
    loss_fn,
    fp16=False,
    scaler=None,
    device=torch.device("cuda"),
):
    tracker = Tracker()

    for sp_input, targets, mask, fn in tqdm(
        get_batch(train_loader, device, fp16),
        leave=False,
    ):
        preds = model.forward(sp_input)

        loss = loss_fn(preds, targets, mask)

        if fp16:
            assert sum(loss.values()).dtype is torch.float32
            scaler.scale(sum(loss.values())).backward()
            scaler.step(optimizer)
            scaler.update()
            scale = scaler.get_scale()
        else:
            (sum(loss.values())).backward()
            optimizer.step()

        optimizer.zero_grad()
        tracker.update(loss)

    return tracker, scaler


@torch.no_grad()
def eval_epoch(
    data_loader,
    model,
    loss_fn,
    fp16=False,
    device=torch.device("cuda"),
):
    tracker = Tracker()
    model.eval()

    for sp_input, targets, mask, fn in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Evaluating",
        leave=False,
    ):
        preds = model.forward(sp_input)
        loss = loss_fn(preds, targets, mask)
        tracker.update(loss)
    model.train()

    return tracker


@torch.no_grad()
def capture_output(
    renderer,
    data_loader,
    model,
    cmap,
    fp16=False,
    device=torch.device("cuda"),
):
    model.eval()
    captures = []

    for sp_input, targets, mask, fn in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Capturing Outputs",
        leave=False,
    ):
        model_output = model.forward(sp_input)

        labelled_clouds = model_output_to_labelled_clds(
            sp_input,
            model_output,
            cmap,
            fn,
        )
        captures.append(
            flatten_list(
                [capture_labelled_cloud(renderer, cld) for cld in labelled_clouds]
            )
        )
    model.train()
    return captures


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )
    run_dir = HydraConfig.get().runtime.output_dir
    run_name = wandb.run.name
    log.info(f"Directory : {run_dir}")
    log.info(f"Machine: {os.uname()[1]}")

    renderer = Renderer(1920, 1080)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()
    torch.save(model, f"{run_dir}/{run_name}_model.pt")

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    loss_fn = instantiate(cfg.loss_fn)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    epochs_no_improve = 0
    best_val_loss = torch.inf

    # Epochs
    for epoch in tqdm(range(0, cfg.num_epoch), leave=True, desc="Epoch"):
        with amp_ctx:
            training_tracker, scaler = train_epoch(
                train_loader,
                model,
                optimizer,
                loss_fn,
                scaler=scaler,
                fp16=cfg.fp16,
            )

            val_tracker = eval_epoch(
                val_loader,
                model,
                loss_fn,
                fp16=cfg.fp16,
            )

            # if (epoch + 1) % cfg.capture_output == 0:
            #     batch_images = capture_output(
            #         renderer,
            #         test_loader,
            #         model,
            #         cfg.cmap,
            #         fp16=cfg.fp16,
            #     )
            #     log_images("Test Output", flatten_list(batch_images), epoch)

        scheduler.step(val_tracker.total_loss) if cfg.lr_decay else None

        # Save Best Model
        print(val_tracker.total_loss)
        if val_tracker.total_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_tracker.total_loss
            wandb.run.summary["Best Test Loss"] = best_val_loss
            torch.save(model.state_dict(), f"{run_dir}/{run_name}_model_weights.pt")
            log.info(f"Weights Saved at epoch: {epoch}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.early_stop_epoch and cfg.early_stop:
            log.info("Training Ended (Evaluation Test Score Not Improving)")
            break

        # log onto wandb...
        training_tracker.log("Training", epoch)
        val_tracker.log("Validation", epoch)


if __name__ == "__main__":
    main()
