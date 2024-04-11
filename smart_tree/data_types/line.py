from dataclasses import dataclass
from typing import List

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base import Base

patch_typeguard()


@typechecked
@dataclass
class LineSegment(Base):
    a: TensorType[3, float]
    b: TensorType[3, float]


@typechecked
@dataclass
class CollatedLineSegment(Base):
    a: TensorType["N", 3, float]
    b: TensorType["N", 3, float]


def collates_line_segments(lines: List[LineSegment]) -> CollatedLineSegment:
    a = torch.cat([l.a for l in lines]).reshape(-1, 3)
    b = torch.cat([l.b for l in lines]).reshape(-1, 3)

    return CollatedLineSegment(a, b)
