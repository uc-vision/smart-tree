import functools
import spconv.pytorch as spconv
import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F

from smart_tree.model.model_blocks import MLP, UBlock, SubMConvBlock
from smart_tree.util.math.maths import torch_normalized

spconv.constants.SPCONV_ALLOW_TF32 = True

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
        branch_classes=[0],
        algo=spconv.ConvAlgo.Native,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.branch_classes = torch.tensor(branch_classes, device=device)

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4)  # , momentum=0.99)
        activation_fn = nn.ReLU

        self.radius_loss = nn.L1Loss()
        self.direction_loss = nn.CosineSimilarity()

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

        self.radius_head = MLP(radius_fc_planes, norm_fn, activation_fn)
        self.direction_head = MLP(direction_fc_planes, norm_fn, activation_fn)
        self.class_head = MLP(class_fc_planes, norm_fn, activation_fn)

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input):
        x = self.input_conv(input)
        unet_out = self.UNet(x)

        radius = self.radius_head(unet_out).features
        direction = self.direction_head(unet_out).features
        class_l = self.class_head(unet_out).features

        return torch.cat(
            [radius, direction, class_l],
            dim=1,
        )

    # @force_fp32(apply_to=("outputs", "targets"))
    def compute_loss(self, outputs, targets, mask=None):
        losses = {}

        if mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

        radius_pred = outputs[:, [0]]
        direction_pred = F.normalize(outputs[:, 1:4])
        class_pred = outputs[:, 4:]

        class_target = targets[:, [3]]
        direction_target, radius_target = torch_normalized(targets[:, :3])

        mask = torch.isin(
            class_target,
            self.branch_classes,
        )

        mask = mask.reshape(-1)

        losses["radius"] = self.compute_radius_loss(
            radius_pred[mask], radius_target[mask]
        )
        losses["direction"] = self.compute_direction_loss(
            direction_pred[mask], direction_target[mask]
        )
        losses["class"] = self.compute_class_loss(class_pred, class_target)

        return losses

    @force_fp32(apply_to=("outputs", "targets"))
    def compute_radius_loss(self, outputs, targets):
        return self.radius_loss(outputs, torch.log(targets))

    @force_fp32(apply_to=("outputs", "targets"))
    def compute_direction_loss(self, outputs, targets):
        return torch.mean(1 - self.direction_loss(outputs, targets))

    @force_fp32(apply_to=("outputs", "targets"))
    def compute_class_loss(self, outputs, targets):
        return self.dice_loss(outputs, targets.long())

    def dice_loss(self, outputs, targets):
        # https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
        smooth = 1
        outputs = F.softmax(outputs, dim=1)
        targets = F.one_hot(targets).reshape(-1, 1)

        intersection = (outputs * targets).sum()

        return 1 - (
            (2.0 * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        )

    def focal_loss(self, outputs, targets):
        # https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py

        gamma = 2
        input = outputs

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -1 * (1 - pt) ** gamma * logpt
        # return loss.sum()
        return loss.mean()

    def nll_loss(self, outputs, targets):
        weights = targets.shape[0] / (torch.bincount(targets))  # Balance class weights
        return F.nll_loss(F.log_softmax(outputs, dim=1), targets, weight=weights)

    def dice_and_focal_loss(self, outputs, targets):
        return self.focal_loss(outputs, targets) + self.dice_loss(outputs, targets)

    # add smoothing loss: https://github.com/guochengqian/openpoints/blob/ee100c81b1d9603c0fc76a3ee4e37d10b2af60ba/loss/cross_entropy.py

    def print_gradients(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                print(m.weight)
                print(m.bias)
