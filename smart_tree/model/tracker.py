import wandb
import numpy as np

# Should be more generalizable ...


class Tracker:
    def __init__(self):
        self.losses = {}

    def update(self, loss_dict: dict):
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

        wandb.log(log_dict, epoch)
