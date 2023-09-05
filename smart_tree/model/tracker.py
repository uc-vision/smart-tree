import torch
import wandb
from sklearn.metrics import f1_score, mean_absolute_percentage_error


class Tracker:
    def __init__(self):
        self.losses = {}
        self.metrics = {}

    def update_losses(self, loss_dict: dict):
        for k, v in loss_dict.items():
            self.losses[k] = self.losses.get(k, []) + [v.item()]

    @property
    def total_loss(self):
        loss = 0
        for k, v in self.losses.items():
            loss += v[-1]
        return loss

    def log(self, name, epoch):
        log_dict = {}

        for k, v in self.losses.items():
            log_dict[k] = v[-1]

        log_dict.update(self.metrics)

        wandb.log(log_dict, epoch)

    @torch.no_grad()
    def update_metrics(self, outputs, targets, mask):
        target_class = targets[:, [-1]].long().cpu()
        predicted_class = torch.argmax(outputs["class_l"], dim=1).cpu()

        if (
            not torch.isnan(predicted_class).all()
            and torch.isfinite(predicted_class).all()
        ):
            self.metrics["f1"] = f1_score(
                target_class.view(-1),
                predicted_class.view(-1),
                average="macro",
            )

        target_radius = targets[:, [0]].cpu()
        predicted_radius = torch.exp(outputs["radius"].float().cpu())
        if (
            not torch.isnan(predicted_radius).all()
            and torch.isfinite(predicted_radius).all()
        ):
            self.metrics["radius_mape"] = mean_absolute_percentage_error(
                target_radius.view(-1),
                predicted_radius.view(-1),
            )
