import wandb
import torch

from sklearn.metrics import f1_score


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

    def update_metrics(self, outputs, targets, mask):
        target_class = targets[:, [-1]].long()
        pred_class = torch.argmax(outputs["class_l"])

        print(torch.argmax(outputs["class_l"], dim=0))
        # print(target_class.shape)
        # print(pred_class.shape)

        quit(0)

        self.metrics["f1"] = f1_score(
            target_class.cpu().view(-1),
            outputs.cpu().view(-1),
            average="macro",
        )
