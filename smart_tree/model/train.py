import contextlib
import logging
import os
from typing import Callable, List

import hydra
import numpy as np
import taichi as ti
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from smart_tree.data_types.cloud import LabelledCloud

from .helper import log_cloud_on_wandb


def train_epoch(data_loader, model, optimizer, scaler=None, tracker=None):
    data_loader_tqdm = tqdm(data_loader, desc="Training", leave=False)

    for model_input, targets, meta_data in data_loader_tqdm:
        preds = model.forward(model_input)
        loss = model.compute_loss(preds, targets, meta_data)

        if scaler:
            assert sum(loss.values()).dtype is torch.float32
            scaler.scale(sum(loss.values())).backward()
            scaler.step(optimizer)
            scaler.update()
            scale = scaler.get_scale()
        else:
            sum(loss.values()).backward()
            optimizer.step()

        optimizer.zero_grad()

        if tracker:
            tracker.update_losses(loss)
            tracker.update_metrics(preds, targets)

    return tracker, scaler


@torch.no_grad()
def eval_epoch(data_loader, model, tracker=None):
    model.eval()

    data_loader_tqdm = tqdm(data_loader, desc="Evaluating", leave=False)

    for model_input, targets, data in data_loader_tqdm:
        preds = model.forward(model_input)
        loss = model.compute_loss(preds, targets, data)

        if tracker:
            tracker.update_losses(loss)
            tracker.update_metrics(preds, targets)

    model.train()

    return tracker


@torch.no_grad()
def capture_clouds(data_loader, model, capture_func: Callable):
    model.eval()
    clouds: List[LabelledCloud] = []

    for model_input, targets, data in data_loader:
        preds = model.forward(model_input)

        clouds += capture_func(model_input, preds, data)

    model.train()
    return clouds


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    ti.init(arch=ti.gpu)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
        dir=cfg.wandb.dir,
    )
    run_dir = HydraConfig.get().runtime.output_dir
    run_name = wandb.run.name
    log.info(f"Directory : {run_dir}")
    log.info(f"Machine: {os.uname()[1]}")
    log.info(f"FP-16: {cfg.fp16}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader)

    training_tracker = instantiate(cfg.training_tracker)
    val_tracker = instantiate(cfg.val_tracker)
    test_tracker = instantiate(cfg.test_tracker)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    if cfg.capture_output:
        capture_function = instantiate(cfg.capture_fn)
        capturer_loader = instantiate(cfg.capture_data_loader)
        log.info(f"Capture Dataset Size: {len(capturer_loader.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()
    torch.save(model, f"{run_dir}/{run_name}_model.pt")

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if cfg.fp16 else None

    epochs_no_improve = 0
    best_epoch_loss = torch.inf
    best_epoch = 0

    # Epochs
    for epoch in tqdm(range(0, cfg.num_epoch), leave=True, desc="Epoch"):
        with amp_ctx:
            training_tracker, scaler = train_epoch(
                train_loader,
                model,
                optimizer,
                scaler=scaler,
                tracker=training_tracker,
            )

            val_tracker = eval_epoch(val_loader, model, val_tracker)
            test_tracker = eval_epoch(test_loader, model, test_tracker)

        if val_tracker.epoch_loss < best_epoch_loss:
            epochs_no_improve = 0
            best_epoch_loss = val_tracker.epoch_loss
            best_epoch = epoch
            wandb.run.summary["Best Val Loss"] = best_epoch_loss
            torch.save(model.state_dict(), f"{run_dir}/{run_name}_model_weights.pt")
            log.info(f"Weights Saved at epoch: {epoch}")
            clouds: List = capture_clouds(capturer_loader, model, capture_function)

        else:
            epochs_no_improve += 1

        # if (
        #     cfg.capture_output and (epoch + 1) % cfg.capture_epoch == 0
        # ) or epochs_no_improve == 0:
        #     if cfg.capture_delete_old_artifacts and artefacts_logged:
        #         for artifact in run.logged_artifacts():
        #             artifact.delete(delete_aliases=True)

        #     artefacts_logged = True

        scheduler.step(val_tracker.epoch_loss) if cfg.lr_decay else None

        if epochs_no_improve == cfg.early_stop_epoch and cfg.early_stop:
            log.info("Training Ended (Evaluation Test Score Not Improving)")
            break

        if cfg.wandb.mode == "disabled":
            log.info(f"Training: {training_tracker}")
            log.info(f"Testing: {test_tracker}")
        else:
            training_tracker.log("Training", epoch)
            test_tracker.log("Testing", epoch)
            val_tracker.log("Validation", epoch)

    log_cloud_on_wandb(clouds, epoch)


if __name__ == "__main__":
    main()
