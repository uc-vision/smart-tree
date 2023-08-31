from __future__ import annotations

from dataclasses import dataclass

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
@dataclass
class Ellipsoid:
    semi_axis_lengths: TensorType[3, 1]
    rotation_matrix: TensorType[3, 3]
    translation: TensorType[3, 3]

    def to_o3d_mesh(self):
        pass
