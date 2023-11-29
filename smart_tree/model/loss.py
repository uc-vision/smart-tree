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
        targets = targets.view(-1, 1).long()

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


# A work-around could be to assume that you have gaussian noise and make the Neural Network predict a mean μ
#  and variance σ
# . For the cost function you can use the NLPD (negative log probability density). For datapoint (xi,yi)
#  that will be −logN(yi−μ(xi),σ(xi))
# . This will make your μ(xi)
#  try to predict your yi
#  and your σ(xi)
#  be smaller when you have more confidence and bigger when you have less.

# To check how good are your assumptions for the validation data you may want to look at yi−μ(xi)σ(xi)
#  to see if they roughly follow a N(0,1)
# . On test data you again want to maximize the probability of your test data so you can use NLPD metric again.
