import functools
from typing import List

import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from smart_tree.model.model_blocks import MLP, ResidualBlock, UBlock


class Smart_Tree(nn.Module):
    def __init__(
        self,
        input_channels: int,
        unet_planes: List[int],
        mlp_layers=2,
        num_classes=1,
        bias=False,
        algo=spconv.ConvAlgo.Native,
    ):
        super().__init__()

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

        # Three Heads...
        self.radius_head = MLP(unet_planes[0], 1, norm_fn, num_layers=mlp_layers)
        self.medial_dir_head = MLP(unet_planes[0], 3, norm_fn, num_layers=mlp_layers)
        self.class_head = MLP(
            unet_planes[0],
            num_classes,
            norm_fn,
            num_layers=mlp_layers,
        )

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

        output_feats = output.features

        predictions["radius"] = self.radius_head(output_feats)
        predictions["medial_direction"] = F.normalize(
            self.medial_dir_head(output_feats)
        )
        predictions["class_l"] = self.class_head(output_feats)

        return predictions


# class Smarter_Tree(nn.Module):
#     def __init__(
#         self,
#         input_channels,
#         unet_planes,
#         radius_fc_planes,
#         medial_direction_fc_planes,
#         branch_direction_fc_planes,
#         class_fc_planes,
#         bias=False,
#         algo=spconv.ConvAlgo.Native,
#     ):
#         super().__init__()

#         norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
#         activation_fn = nn.ReLU

#         self.input_conv = SubMConvBlock(
#             input_channels=input_channels,
#             output_channels=unet_planes[0],
#             kernel_size=1,
#             padding=1,
#             norm_fn=norm_fn,
#             activation_fn=activation_fn,
#         )

#         self.UNet = UBlock(
#             unet_planes,
#             norm_fn,
#             activation_fn,
#             key_id=1,
#             algo=algo,
#         )

#         # Three Heads...
#         self.radius_head = MLP(
#             radius_fc_planes,
#             norm_fn,
#             activation_fn,
#             bias=True,
#         )
#         self.medial_direction_head = MLP(
#             medial_direction_fc_planes,
#             norm_fn,
#             activation_fn,
#             bias=True,
#         )

#         self.branch_direction_head = MLP(
#             branch_direction_fc_planes,
#             norm_fn,
#             activation_fn,
#             bias=True,
#         )

#         self.class_head = MLP(
#             class_fc_planes,
#             norm_fn,
#             activation_fn,
#             bias=True,
#         )

#         self.apply(self.set_bn_init)

#     @staticmethod
#     def set_bn_init(m):
#         classname = m.__class__.__name__
#         if classname.find("BatchNorm") != -1:
#             m.weight.data.fill_(1.0)
#             m.bias.data.fill_(0.0)

#     def forward(self, input):
#         predictions = {}

#         x = self.input_conv(input)
#         unet_out = self.UNet(x)

#         predictions["radius"] = self.radius_head(unet_out).features
#         predictions["medial_direction"] = F.normalize(
#             self.medial_direction_head(unet_out).features
#         )
#         predictions["branch_direction"] = F.normalize(
#             self.branch_direction_head(unet_out).features
#         )

#         predictions["class_l"] = self.class_head(unet_out).features

#         return predictions
