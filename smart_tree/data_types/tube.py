from dataclasses import asdict, dataclass
from typing import List

import torch
from torchtyping import TensorType
from typeguard import typechecked


@typechecked
@dataclass
class Tube:
    a: TensorType[3, float]
    b: TensorType[3, float]
    r1: TensorType[1, float]
    r2: TensorType[1, float]

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            args[k] = v.to(device)

        return Tube(**args)


@typechecked
@dataclass
class CollatedTube:
    a: TensorType["N", 3, float]
    b: TensorType["N", 3, float]
    r1: TensorType["N", 1, float]
    r2: TensorType["N", 1, float]

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            args[k] = v.to(device)

        return Tube(**args)


def collate_tubes(tubes: List[Tube]) -> CollatedTube:
    a = torch.stack([tube.a for tube in tubes])
    b = torch.stack([tube.b for tube in tubes])
    r1 = torch.stack([tube.r1 for tube in tubes])
    r2 = torch.stack([tube.r2 for tube in tubes])

    return CollatedTube(a, b, r1, r2)
