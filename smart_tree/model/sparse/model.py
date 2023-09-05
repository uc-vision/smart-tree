import functools
from typing import List

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss import DirectionLoss, FocalLoss
from .model_blocks import MLP, ResidualBlock, UBlock


class Smart_Tree(nn.Module):
    def __init__(
        self,
        input_channels: int,
        unet_planes: List[int],
        mlp_layers=2,
        num_classes=1,
        log_radius=True,
    ):
        super().__init__()

        self.class_loss = FocalLoss()
        self.radius_loss = nn.L1Loss()
        self.direction_loss = DirectionLoss()
        self.log_radius = log_radius

        norm_fn = functools.partial(
            nn.BatchNorm1d,
            eps=1e-4,
            momentum=0.1,
        )

        self.input_conv = spconv.SubMConv3d(
            input_channels,
            unet_planes[0],
            kernel_size=3,
            padding=1,
            bias=False,
            algo=spconv.ConvAlgo.Native,
        )

        self.unet = UBlock(
            unet_planes,
            norm_fn,
            block_reps=2,
            indice_key_id=1,
            block=ResidualBlock,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(unet_planes[0]), nn.ReLU())

        self.radius_head = MLP(unet_planes[0], 1, norm_fn, mlp_layers)
        self.medial_dir_head = MLP(unet_planes[0], 3, norm_fn, mlp_layers)
        self.class_head = MLP(unet_planes[0], num_classes, norm_fn, mlp_layers)

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, sparse_input):
        predictions = {}

        output = self.input_conv(sparse_input)
        output = self.unet(output)
        output = self.output_layer(output)

        self.output_feats = output.features

        predictions["radius"] = self.radius_head(self.output_feats)
        predictions["medial_direction"] = F.normalize(
            self.medial_dir_head(self.output_feats)
        )
        predictions["class_l"] = self.class_head(self.output_feats)

        return predictions

    def compute_loss(
        self,
        predictions: dict,
        targets: torch.tensor,
        mask: torch.tensor,
    ):
        losses = {}

        mask = mask.reshape(-1)

        target_class = targets[:, [-1]].long()
        pred_class = predictions["class_l"]

        target_radius = targets[:, [0]]
        if self.log_radius:
            target_radius = torch.log(target_radius)

        pred_radius = predictions["radius"]

        target_direction = targets[:, 1:4]
        pred_direction = predictions["medial_direction"]

        losses["class_l"] = self.class_loss(pred_class[mask], target_class[mask])
        losses["radius"] = self.radius_loss(pred_radius[mask], target_radius[mask])
        losses["medial_direction"] = self.direction_loss(
            pred_direction[mask], target_direction[mask]
        )

        return losses


class Smarter_Tree(Smart_Tree):
    def __init__(
        self,
        input_channels: int,
        unet_planes: List[int],
        mlp_layers=2,
        num_classes=1,
    ):
        super().__init__(input_channels, unet_planes, mlp_layers, num_classes)
        norm_fn = functools.partial(
            nn.BatchNorm1d,
            eps=1e-4,
            momentum=0.1,
        )

        self.branch_direction_head = MLP(unet_planes[0], 3, norm_fn, mlp_layers)

    def forward(self, sparse_input):
        predictions = super().forward(sparse_input)

        predictions["branch_direction"] = F.normalize(
            self.branch_direction_head(self.output_feats)
        )

        return predictions

    def compute_loss(
        self,
        predictions: dict,
        targets: torch.tensor,
        mask: torch.tensor,
    ):
        losses = super().compute_loss(predictions, targets, mask)

        mask = mask.reshape(-1)
        target_direction = targets[:, 4:7]
        pred_direction = predictions["branch_direction"]
        losses["branch_direction"] = self.direction_loss(
            pred_direction[mask],
            target_direction[mask],
        )

        return losses
