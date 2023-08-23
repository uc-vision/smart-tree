import contextlib
import logging
import math
import os
from pathlib import Path
from typing import List

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from smart_tree.data_types.cloud import LabelledCloud
from smart_tree.model.render import RenderQueue
from smart_tree.util.visualizer.camera import Renderer

import wandb
import taichi as ti
from smart_tree.util.misc import flatten_list

from smart_tree.model.helper import get_batch, model_output_to_labelled_clds
from smart_tree.model.tracker import Tracker


def train_epoch(
    train_loader,
    model,
    optimizer,
    fp16=False,
    scaler=None,
    device=torch.device("cuda"),
):
    tracker = Tracker()

    for sp_input, targets, mask, filenames in tqdm(
        get_batch(train_loader, device, fp16),
        desc="Training",
        leave=False,
    ):
        output = model.forward(sp_input)
        loss = model.compute_loss(output, targets, mask)

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
    fp16=False,
    device=torch.device("cuda"),
):
    tracker = Tracker()
    model.eval()

    for sp_input, targets, mask, filenames in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Evaluating",
        leave=False,
    ):
        output = model.forward(sp_input)
        loss = model.compute_loss(output, targets, mask)
        tracker.update(loss)
    model.train()

    return tracker


@torch.no_grad()
def capture_clouds(
    data_loader,
    model,
    cmap,
    fp16=False,
    device=torch.device("cuda"),
) -> List[LabelledCloud]:
    model.eval()
    clouds = []

    for sp_input, targets, mask, filenames in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Capturing Outputs",
        leave=False,
        total=math.ceil(len(data_loader) / data_loader.batch_size),
    ):
        model_output = model.forward(sp_input)
        clouds.extend(
            model_output_to_labelled_clds(sp_input, model_output, cmap, filenames)
        )

    model.train()
    return clouds


def capture_and_log(loader, model, epoch, wandb_run, cfg):
    clouds = capture_clouds(
        loader,
        model,
        cfg.cmap,
        fp16=cfg.fp16,
    )

    for cloud in clouds:
        xyz_rgb = torch.cat((cloud.xyz, cloud.cmap[cloud.class_l] * 255), -1)
        wandb_run.log(
            {f"{Path(cloud.filename).stem}": wandb.Object3D(xyz_rgb.numpy())},
            step=epoch,
        )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    cfg = cfg.training

    ti.init(arch=ti.gpu)

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )

    run_dir = HydraConfig.get().runtime.output_dir
    run_name = wandb.run.name

    log.info(f"Directory : {run_dir}")
    log.info(f"Machine: {os.uname()[1]}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method("spawn")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()

    if cfg.get("weights_path", None) is not None:
        print(f"Loading weights from {cfg.weights_path}")

        weights = torch.load(cfg.weights_path, map_location=device)
        model.load_state_dict(weights)

    torch.save(model, f"{run_dir}/{run_name}_model.pt")
    # render_queue = RenderQueue(image_size=(960, 540), wandb_run=wandb_run)

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    epochs_no_improve = 0
    best_val_loss = torch.inf

    # Epochs
    for epoch in tqdm(range(0, cfg.num_epoch), leave=True):
        with amp_ctx:
            val_tracker = eval_epoch(
                val_loader,
                model,
                fp16=cfg.fp16,
            )

            training_tracker, scaler = train_epoch(
                train_loader,
                model,
                optimizer,
                scaler=scaler,
                fp16=cfg.fp16,
            )

            if (epoch + 1) % cfg.capture_output == 0:
                capture_and_log(test_loader, model, epoch, wandb_run, cfg)
                # capture_and_log(train_loader, model, epoch, wandb_run, cfg)

        scheduler.step(val_tracker.total_loss) if cfg.lr_decay else None

        # Save Best Model
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
