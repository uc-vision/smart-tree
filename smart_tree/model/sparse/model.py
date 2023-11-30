import functools
from typing import Dict, List, Tuple
from dataclasses import asdict

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.utils import gather_features_by_pc_voxel_id

from ..loss import DirectionLoss, FocalLoss
from .model_blocks import MLP, ResidualBlock, UBlock
from .util import Data


class Smart_Tree(nn.Module):
    def __init__(
        self,
        input_channels: int,
        unet_planes: List[int],
        mlp_layers=2,
        num_classes=1,
        log_radius=True,
        target_features=[],
    ):
        super().__init__()

        self.target_features = target_features

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

        if "radius" in self.target_features:
            self.radius_head = MLP(unet_planes[0], 1, norm_fn, mlp_layers)
        if "medial_direction" in self.target_features:
            self.medial_dir_head = MLP(unet_planes[0], 3, norm_fn, mlp_layers)
        if "class_l" in self.target_features:
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

        if "radius" in self.target_features:
            predictions["radius"] = self.radius_head(self.output_feats)

        if "medial_direction" in self.target_features:
            predictions["medial_direction"] = F.normalize(
                self.medial_dir_head(self.output_feats)
            )
        if "class_l" in self.target_features:
            predictions["class_l"] = self.class_head(self.output_feats)

        return predictions

    def compute_loss(
        self,
        predictions: dict,
        targets: torch.tensor,
        mask: torch.tensor,
    ):
        losses = {}

        mask = mask.reshape(-1).long()

        if "radius" in self.target_features:
            target_radius = targets["radius"][mask]
            if self.log_radius:
                target_radius = torch.log(target_radius)
            pred_radius = predictions["radius"][mask]
            losses["radius"] = self.radius_loss(pred_radius, target_radius)

        if "medial_direction" in self.target_features:
            tgt_dir = targets["medial_direction"][mask]
            pred_dir = predictions["medial_direction"][mask]
            losses["medial_direction"] = self.direction_loss(pred_dir, tgt_dir)

        if "class_l" in self.target_features:
            target_class = targets["class_l"][mask]
            pred_class = predictions["class_l"][mask]
            losses["class_l"] = self.class_loss(pred_class, target_class)

        return losses


class Smart_Tree_Self_Consistency(Smart_Tree):
    def __init__(self, *args, confidence_threshold=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf_threshold = confidence_threshold
        self.psuedo_loss_inverse_weight = 0.99

        # self.class_loss = nn.CrossEntropyLoss()

    def forward(self, model_input):
        if isinstance(model_input, tuple):
            pred1 = super().forward(model_input[0])
            pred2 = super().forward(model_input[1])
            return pred1, pred2

        return super().forward(model_input)

    def compute_loss(
        self,
        voxel_predictions: Tuple[Dict],
        voxel_targets: Tuple[Data],
        data,
    ):
        loss = {}

        def gather_pt_data(dict: Dict, data: Data):
            for k, v in dict.items():
                dict[k] = gather_features_by_pc_voxel_id(v, data.voxel_id)
            return dict

        pts1_preds = gather_pt_data(voxel_predictions[0], data[0])
        pts1_targets = gather_pt_data(voxel_targets[0], data[0])

        pts2_preds = gather_pt_data(voxel_predictions[1], data[1])
        pts2_targets = gather_pt_data(voxel_targets[1], data[1])

        data_1_point_ids = data[0].point_id
        data_2_point_ids = data[1].point_id

        mask_1 = data[0].mask.squeeze(1)

        mask_2 = data[1].mask.squeeze(1)

        max_pt_id = max(data_1_point_ids.shape[0], data_2_point_ids.shape[0]) * 2

        pts1_point_ids = data_1_point_ids[:, 0] * max_pt_id + data_1_point_ids[:, 1]
        pts2_point_ids = data_2_point_ids[:, 0] * max_pt_id + data_2_point_ids[:, 1]

        assert pts1_point_ids.shape[0] == torch.unique(pts1_point_ids, dim=0).size(0), (
            pts1_point_ids.shape[0],
            torch.unique(pts1_point_ids, dim=0).size(0),
            "Each point must have a unique ID",
        )
        assert pts2_point_ids.shape[0] == torch.unique(pts2_point_ids, dim=0).size(0), (
            pts2_point_ids.shape[0],
            torch.unique(pts2_point_ids, dim=0).size(0),
            "Each point must have a unique ID",
        )

        valid_pts1_mask = torch.isin(pts1_point_ids, pts2_point_ids)
        valid_pts2_mask = torch.isin(pts2_point_ids, pts1_point_ids)

        assert valid_pts1_mask.sum().item() == valid_pts2_mask.sum().item()

        class_confidence = torch.softmax(pts1_preds["class_l"][valid_pts1_mask], dim=-1)
        max_probs, soft_label_target = torch.max(class_confidence, dim=-1)
        mask_pt = max_probs > self.conf_threshold

        # print(f"Number confident soft labels {mask_pt.sum()}")

        loss["class_loss"] = self.class_loss(
            pts1_preds["class_l"][mask_1],
            pts1_targets["class_l"].squeeze(1)[mask_1],
        )

        loss["class_psuedo_loss"] = (
            self.class_loss(
                pts2_preds["class_l"][valid_pts2_mask][mask_pt],
                soft_label_target[mask_pt],
            )
            * 0.5
        )
        # ) * (1 - self.psuedo_loss_inverse_weight)
        # nan_mask = torch.isnan(loss["class_psuedo_loss"])
        # loss["class_psuedo_loss"][nan_mask] = 0.0

        return loss


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
