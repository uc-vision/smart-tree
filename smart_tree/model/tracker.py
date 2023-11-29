import torch
from sklearn.metrics import f1_score, mean_absolute_percentage_error

import wandb


class Tracker:
    def __init__(self):
        self.losses = {}

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.losses.items())

    def update_losses(self, loss_dict: dict):
        for k, v in loss_dict.items():
            self.losses.setdefault(k, []).append(v.item())

    @property
    def total_loss(self):
        return sum(v[-1] for v in self.losses.values())

    def log(self, name, epoch):
        log_dict = {f"{name}_{k}": v[-1] for k, v in self.losses.items()}
        wandb.log(log_dict, epoch)


class SegmentationTracker(Tracker):
    def __init__(self):
        super().__init__()
        self.metrics = {}

    def __str__(self):
        string_1 = super().__str__()
        string_2 = "\n".join(f"{k}: {v}" for k, v in self.metrics.items())
        return string_1 + "\n" + string_2

    @torch.no_grad()
    def update_metrics(self, outputs, targets):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(targets, tuple):
            targets = targets[0]

        target_class = targets["class_l"].long().cpu()
        predicted_class = torch.argmax(outputs["class_l"], dim=1).cpu()

        self.metrics["f1"] = f1_score(
            target_class.view(-1),
            predicted_class.view(-1),
            average="macro",
        )

        # target_radius = targets[:, [0]].cpu()
        # predicted_radius = torch.exp(outputs["radius"].float().cpu())
        # if (
        #     not torch.isnan(predicted_radius).all()
        #     and torch.isfinite(predicted_radius).all()
        # ):
        #     self.metrics["radius_mape"] = mean_absolute_percentage_error(
        #         target_radius.view(-1),
        #         predicted_radius.view(-1),
        #     )

    def log(self, name, epoch):
        super().log(name, epoch)
        log_dict = {f"{name}_{k}": v for k, v in self.metrics.items()}
        wandb.log(log_dict, epoch)
