import functools
import spconv.pytorch as spconv
import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F

from smart_tree.model.model_blocks import SparseFC, UBlock, SubMConvBlock
from smart_tree.util.maths import torch_normalized


from .fp16 import force_fp32


class Smart_Tree(nn.Module):
    def __init__(
        self,
        input_channels,
        unet_planes,
        radius_fc_planes,
        direction_fc_planes,
        class_fc_planes,
        bias=False,
        algo=spconv.ConvAlgo.Native,
    ):
        super().__init__()

        norm_fn = functools.partial(
            nn.BatchNorm1d,
            eps=1e-4,
            momentum=0.1,
        )
        activation_fn = nn.ReLU

        self.input_conv = SubMConvBlock(
            input_channels=input_channels,
            output_channels=unet_planes[0],
            kernel_size=1,
            padding=1,
            norm_fn=norm_fn,
            activation_fn=activation_fn,
        )

        self.UNet = UBlock(
            unet_planes,
            norm_fn,
            activation_fn,
            key_id=1,
            algo=algo,
        )

        # Three Heads...
        self.radius_head = SparseFC(
            radius_fc_planes,
            norm_fn,
            activation_fn,
            algo=algo,
        )
        self.direction_head = SparseFC(
            direction_fc_planes,
            norm_fn,
            activation_fn,
            algo=algo,
        )
        self.class_head = SparseFC(
            class_fc_planes,
            norm_fn,
            activation_fn,
            algo=algo,
        )

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input):
        predictions = {}

        x = self.input_conv(input)
        unet_out = self.UNet(x)

        predictions["radius"] = self.radius_head(unet_out).features
        predictions["direction"] = F.normalize(self.direction_head(unet_out).features)
        predictions["class_l"] = self.class_head(unet_out).features

        return predictions
