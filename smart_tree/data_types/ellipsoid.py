from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, rand
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import o3d_viewer
from ..util.misc import to_torch, voxel_downsample
from ..util.queries import skeleton_to_points

patch_typeguard()


@typechecked
@dataclass
class Ellipsoid:
    semi_axis_lengths: TensorType[3, 1]
    rotation_matrix: TensorType[3, 3]
    translation: TensorType[3, 3]

    def to_o3d_mesh(self):
        pass
