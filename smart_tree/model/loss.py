import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import patch_typeguard

patch_typeguard()


class FocalLoss(nn.Module):
    # https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py

    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        if outputs.dim() > 2:
            outputs = outputs.view(
                outputs.size(0), outputs.size(1), -1
            )  # N,C,H,W => N,C,H*W
            outputs = outputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            outputs = outputs.contiguous().view(
                -1, outputs.size(2)
            )  # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(outputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DirectionLoss(nn.Module):
    def __init__(self):
        super(DirectionLoss, self).__init__()
        self.loss_fn = nn.CosineSimilarity()

    def forward(self, outputs, targets):
        return torch.mean(1 - self.loss_fn(outputs, targets))
