import contextlib
import logging
import os
from functools import partial

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from py_structs.torch import map_tensors
from tqdm import tqdm

import wandb
from smart_tree.dataset.dataset import TreeDataset
from smart_tree.model.model import Smart_Tree
from smart_tree.model.sparse import batch_collate, sparse_from_batch
from smart_tree.util.mesh.geometries import o3d_cloud
from smart_tree.util.misc import concate_dict_of_tensors
from smart_tree.util.visualizer.camera import o3d_headless_render
from smart_tree.util.math.maths import torch_normalized
from sklearn.metrics import f1_score, mean_absolute_percentage_error


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="tree-dataset",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    log = logging.getLogger(__name__)

    cfg = cfg.training

    # Init
    wandb.init(
        project=cfg.name,
        entity="harry1576",
        mode=cfg.wblogging,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )

    log.info(
        f"Directory : {HydraConfig.get().runtime.output_dir}, Machine: {os.uname()[1]}"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataloader = instantiate(
        cfg.data_loader,
        dataset=instantiate(cfg.dataset, mode="train"),
    )
    val_dataloader = instantiate(
        cfg.data_loader,
        drop_last=False,
        dataset=instantiate(cfg.dataset, mode="validation"),
    )
    log.info(f"Train Dataset Size: {len(train_dataloader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_dataloader.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()
    torch.save(
        model,
        f"{HydraConfig.get().runtime.output_dir}/{wandb.run.name}_model.pt",
    )

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler(
        growth_factor=1.1, backoff_factor=0.90, growth_interval=1
    )
    clip = 1

    epochs_no_improve = 0
    best_val_loss = torch.inf
    num_input_feats = 6 if cfg.use_colour else 3

    # Training Epochs
    for epoch in tqdm(range(0, cfg.num_epoch), leave=False):
        epoch_training_loss = {"radius": 0.0, "direction": 0.0, "class": 0.0}

        # Training Loop
        for features, coordinates, loss_mask in tqdm(
            train_dataloader,
            leave=False,
            desc="Training...",
        ):
            with amp_ctx:
                sparse_input = sparse_from_batch(
                    features[:, :num_input_feats],
                    coordinates,
                    device=device,
                )
                targets = map_tensors(
                    features[:, 6:10],
                    partial(
                        torch.Tensor.to,
                        device=device,
                    ),
                )

                model_output = model.forward(sparse_input)

                loss = model.compute_loss(model_output, targets, loss_mask)
                total_loss = loss["radius"] + loss["direction"] + loss["class"]

            if cfg.fp16:
                scaler.scale(
                    loss["radius"] + loss["direction"] + loss["class"]
                ).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            epoch_training_loss["radius"] += loss["radius"].item()
            epoch_training_loss["direction"] += loss["direction"].item()
            epoch_training_loss["class"] += loss["class"].item()

        # Validation Loop
        with torch.no_grad():
            epoch_validation_loss = {"radius": 0.0, "direction": 0.0, "class": 0.0}
            val_preds = {"radius": [], "direction": [], "class": []}
            val_targets = {"radius": [], "direction": [], "class": []}

            for features, coordinates, loss_mask in tqdm(
                val_dataloader,
                leave=False,
                desc="Validating",
            ):
                sparse_input = sparse_from_batch(
                    features[:, :num_input_feats],
                    coordinates,
                    device=device,
                )
                targets = map_tensors(
                    features[:, 6:10],
                    partial(torch.Tensor.to, device=device),
                )

                with amp_ctx:
                    model_output = model.forward(sparse_input)
                    loss = model.compute_loss(model_output, targets, loss_mask)

                val_preds["radius"].append(torch.exp(model_output[:, [0]]))
                val_preds["direction"].append(model_output[:, 1:4])
                val_preds["class"].append(torch.argmax(model_output[:, 4:], dim=1))

                direction_target, radius_target = torch_normalized(targets[:, :3])
                val_targets["radius"].append(radius_target.squeeze(1))
                val_targets["direction"].append(direction_target)
                val_targets["class"].append(targets[:, 3])

                epoch_validation_loss["radius"] += loss["radius"]
                epoch_validation_loss["direction"] += loss["direction"]
                epoch_validation_loss["class"] += loss["class"]

            val_los = sum(epoch_validation_loss.values())
            val_preds = concate_dict_of_tensors(val_preds)
            val_targets = concate_dict_of_tensors(val_targets)

            scheduler.step(val_los) if cfg.lr_decay else None

        if val_los < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_los

            wandb.run.summary["Best Test Loss"] = best_val_loss

            torch.save(
                model.state_dict(),
                f"{HydraConfig.get().runtime.output_dir}/{wandb.run.name}_model_weights.pt",
            )
            log.info(f"Weights Saved at epoch: {epoch}")

            batch_filter = coordinates[:, 0] == 0  # First Item ...
            xyz = features[:, :3][batch_filter].to(torch.device("cpu"))
            output = model_output[batch_filter].to(torch.device("cpu"))

            vector = torch.exp(output[:, [0]]) * output[:, 1:4]

            rgb_img = o3d_headless_render(
                [
                    o3d_cloud(xyz, colour=(1, 0, 0)),
                    o3d_cloud(xyz + vector, colour=(0, 1, 0)),
                ],
                camera_position=[3, 0, 0],
                camera_up=[0, 1, 0],
            )

            wandb.log({"Output": wandb.Image(np.asarray(rgb_img))}, step=epoch)

        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.early_stop_epoch and cfg.early_stop:
            log.info("Training Ended (Evaluation Test Score Not Improving)")
            break

        wandb.log(
            {
                "Training Total Loss": (
                    epoch_training_loss["radius"]
                    + epoch_training_loss["direction"]
                    + epoch_training_loss["class"]
                )
                / len(train_dataloader.dataset),
                #
                "Training Vector Loss": (
                    epoch_training_loss["radius"] + epoch_training_loss["direction"]
                )
                / len(train_dataloader.dataset),
                #
                "Training Segmentation Loss": epoch_training_loss["class"]
                / len(train_dataloader.dataset),
                #
                "Validation Total Loss": (
                    epoch_validation_loss["radius"]
                    + epoch_validation_loss["direction"]
                    + epoch_validation_loss["class"]
                )
                / len(val_dataloader.dataset),
                #
                "Validation Vector Loss": (
                    epoch_validation_loss["radius"] + epoch_validation_loss["direction"]
                )
                / len(val_dataloader.dataset),
                #
                "Validation Vector Loss": (
                    epoch_validation_loss["radius"] + epoch_validation_loss["direction"]
                )
                / len(val_dataloader.dataset),
                #
                "Validation Segmentation Loss": epoch_validation_loss["class"]
                / len(val_dataloader.dataset),
                #
                "Validation F1 Score": f1_score(
                    val_preds["class"],
                    val_targets["class"],
                    average="micro",
                ),
                #
                "Validation Radius MAPE": mean_absolute_percentage_error(
                    val_targets["radius"][val_targets["class"] == 0],
                    val_preds["radius"][val_targets["class"] == 0],
                ),
            },
            step=epoch,
        )


if __name__ == "__main__":
    main()
