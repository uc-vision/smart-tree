from dataclasses import dataclass
from typing import List

import torch
from torchtyping import TensorType
from typeguard import typechecked

from .base import Base


@typechecked
@dataclass
class Tube(Base):
    a: TensorType[3, float]
    b: TensorType[3, float]
    r1: TensorType[1, float]
    r2: TensorType[1, float]


@typechecked
@dataclass
class CollatedTube(Base):
    a: TensorType["N", 3, float]
    b: TensorType["N", 3, float]
    r1: TensorType["N", 1, float]
    r2: TensorType["N", 1, float]


def collate_tubes(tubes: List[Tube]) -> CollatedTube:
    a = torch.stack([tube.a for tube in tubes])
    b = torch.stack([tube.b for tube in tubes])
    r1 = torch.stack([tube.r1 for tube in tubes])
    r2 = torch.stack([tube.r2 for tube in tubes])

    return CollatedTube(a, b, r1, r2)
