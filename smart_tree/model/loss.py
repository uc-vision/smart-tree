import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(
    preds,
    targets,
    mask=None,
    radius_loss_fn=None,
    direction_loss_fn=None,
    class_loss_fn=None,
    target_radius_log=True,
    vector_class=None,
):
    losses = {}
    target_class = targets[:, [-1]].long()[mask]
    pred_class = F.softmax(preds["class_l"][mask], dim=1)

    print(preds["class_l"].shape[0])
    print(pred_class.shape[0])
    print(target_class.shape[0])

    quit(0)

    assert target_class.shape[0] == pred_class.shape[0]

    losses["class_l"] = class_loss_fn(pred_class, target_class)

    predicted_radius = preds["radius"]  #
    if target_radius_log:
        target_radius = torch.log(targets[:, [0]])
    else:
        target_radius = targets[:, [0]]

    vector_mask = torch.isin(
        target_class,
        torch.tensor(vector_class, device=target_class.device),
    )[mask].reshape(-1)

    losses["radius"] = radius_loss_fn(
        predicted_radius[vector_mask], target_radius[vector_mask]
    )

    pred_medial_dir = preds["medial_direction"]  #
    target_medial_dir = F.normalize(targets[:, 1:4])
    losses["medial_direction"] = direction_loss_fn(
        pred_medial_dir[vector_mask], target_medial_dir[vector_mask]
    )

    return losses

    # print(preds)

    # match preds:
    #     case {
    #         "radius": radius,
    #         "medial_direction": medial_direction,
    #         "branch_direction": branch_direction,
    #         "class_l": class_l,
    #     }:
    #         predicted_medial_direction = preds["medial_direction"][mask]
    #         predicted_branch_direction = preds["branch_direction"][mask]
    #         target_medial_direction = F.normalize(targets[:, 1:4][mask])
    #         target_branch_direction = F.normalize(targets[:, 4:7][mask])

    #         losses["branch_direction"] = direction_loss_fn(
    #             predicted_branch_direction,
    #             target_branch_direction,
    #         )

    #     case {"radius": radius, "medial_direction": direction, "class_l": class_l}:
    #         predicted_medial_direction = preds["medial_direction"][mask]
    #         target_medial_direction = F.normalize(targets[:, 1:4][mask])

    # predicted_radius = preds["radius"][mask]  #
    # target_radius = targets[:, [0]][mask]

    # predicted_class = preds["class_l"][mask]
    # target_class = targets[:, [-1]].long()[mask]

    # vector_mask = torch.isin(
    #     target_class,
    #     torch.tensor(vector_class, device=target_class.device),
    # ).reshape(-1)

    # if target_radius_log:
    #     target_radius = torch.log(target_radius)

    # losses["medial_direction"] = direction_loss_fn(
    #     predicted_medial_direction[vector_mask], target_medial_direction[vector_mask]
    # )
    # losses["radius"] = radius_loss_fn(
    #     predicted_radius.view(-1)[vector_mask], target_radius.view(-1)[vector_mask]
    # )
    # losses["class_l"] = class_loss_fn(predicted_class, target_class)

    return {1: 1}


def L1Loss(outputs, targets):
    loss = nn.L1Loss()
    return loss(outputs, targets)


def cosine_similarity_loss(outputs, targets):
    loss = nn.CosineSimilarity()
    return torch.mean(1 - loss(outputs, targets))


def dice_loss(output, target, smooth=1.0):
    # Flatten the predictions and targets
    output = output.view(-1)
    target = target.view(-1)

    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)

    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)

    # The Dice Loss is 1 - Dice Coefficient
    dice_loss = 1.0 - dice_coefficient

    return dice_loss


# def focal_loss(outputs, targets):
#     # https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
#     gamma = 2

#     if outputs.dim() > 2:
#         outputs = outputs.view(
#             outputs.size(0), outputs.size(1), -1
#         )  # N,C,H,W => N,C,H*W
#         outputs = outputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
#         outputs = outputs.contiguous().view(-1, outputs.size(2))  # N,H*W,C => N*H*W,C
#     targets = targets.view(-1, 1)
#     logpt = F.log_softmax(outputs, dim=1)
#     logpt = logpt.gather(1, targets)
#     logpt = logpt.view(-1)
#     pt = logpt.exp()
#     loss = -1 * (1 - pt) ** gamma * logpt
#     # return loss.sum()
#     return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def nll_loss(outputs, targets):
    return torch.tensor([0]).cuda()
    weights = targets.shape[0] / (torch.bincount(targets))  # Balance class weights
    return F.nll_loss(F.log_softmax(outputs, dim=1), targets, weight=weights)
