import contextlib
import logging
import os

import hydra
import torch
from beartype.typing import Callable
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
import wandb
from ..o3d_abstractions.camera import Renderer


def train_epoch(
    data_loader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    scaler=None,
    tracker=None,
    device=torch.device("cuda"),
):
    data_loader_tqdm = tqdm(data_loader, desc="Training", leave=False)
    total_loss = 0
    all_preds = []
    all_targets = []

    for model_input, targets in data_loader_tqdm:

        model_input = model_input.to(device)
        targets = targets.to(device)

        preds = model(model_input)

        loss = loss_fn(preds, targets)

        if tracker:
            tracker.update_losses(loss)

        total_loss += sum(loss.values()).item()

        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

        if scaler:
            scaler.scale(sum(loss.values())).backward()
            scaler.step(optimizer)
            scaler.update()
            scale = scaler.get_scale()
        else:
            sum(loss.values()).backward()
            optimizer.step()

        optimizer.zero_grad()

    return tracker, scaler


@torch.no_grad()
def eval_epoch(
    data_loader: DataLoader,
    model: Module,
    loss_fn: Callable,
    tracker=None,
    device=torch.device("cuda"),
):
    model.eval()

    data_loader_tqdm = tqdm(data_loader, desc="Evaluating", leave=False)
    all_preds = []
    all_targets = []

    for model_input, targets in data_loader_tqdm:

        model_input = model_input.to(device)
        targets = targets.to(device)

        preds = model(model_input)

        loss = loss_fn(preds, targets)

        if tracker:
            tracker.update_losses(loss)

        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

    model.train()

    return tracker


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    log = logging.getLogger(__name__)

    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        name=cfg.logging.name,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )
    run_dir = HydraConfig.get().runtime.output_dir
    run_name = wandb.run.name
    log.info(f"Directory : {run_dir}")
    log.info(f"Machine: {os.uname()[1]}")

    renderer = Renderer(960, 540)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders = instantiate(cfg.dataset)
    trackers = instantiate(cfg.tracker)

    log.info(f"Train Dataset Size: {len(data_loaders.train.dataset)}")
    log.info(f"Validation Dataset Size: {len(data_loaders.validation.dataset)}")
    log.info(f"Test Dataset Size: {len(data_loaders.test.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()
    torch.save(model, f"{run_dir}/{run_name}_model.pt")

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    loss_fn = instantiate(cfg.loss)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    epochs_no_improve = 0
    best_val_loss = torch.inf

    # Epochs
    for epoch in (pbar := tqdm(range(0, cfg.num_epoch), leave=True, desc="Epoch")):
        with amp_ctx:
            trackers.train, scaler = train_epoch(
                data_loaders.train,
                model,
                optimizer,
                loss_fn,
                scaler=scaler,
                device=device,
                tracker=trackers.train,
            )

            trackers.validation = eval_epoch(
                data_loaders.validation,
                model,
                loss_fn,
                device=device,
                tracker=trackers.validation,
            )

            trackers.test = eval_epoch(
                data_loaders.test,
                model,
                loss_fn,
                tracker=trackers.test,
            )

            # if (epoch + 1) % cfg.capture_output == 0:
            #     capture_and_log(test_loader, model, epoch, wandb.run, cfg)
            #     capture_and_log(val_loader, model, epoch, wandb.run, cfg)

        scheduler.step(trackers.validation.epoch_loss) if cfg.lr_decay else None

        # pbar.set_description(
        #     f"Epoch {epoch} - Train Loss: {trackers.train.epoch_loss:.4f}, \
        #       Validation Loss: {trackers.validation.epoch_loss:.4f}"
        # )

        # Save Best Model
        if trackers.validation.epoch_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = trackers.validation.epoch_loss
            wandb.run.summary["Best Test Loss"] = best_val_loss
            torch.save(model.state_dict(), f"{run_dir}/{run_name}_model_weights.pt")
            log.info(f"Weights Saved at epoch: {epoch}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.early_stop_epoch and cfg.early_stop:
            log.info("Training Ended (Evaluation Test Score Not Improving)")
            break

        trackers.train.log(epoch).reset()
        trackers.validation.log(epoch).reset()
        trackers.test.log(epoch).reset()


if __name__ == "__main__":
    main()
