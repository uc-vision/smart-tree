from dataclasses import asdict, dataclass
from typing import List

import torch
from torchtyping import TensorType
from typeguard import typechecked


@typechecked
@dataclass
class LineSegment:
    a: TensorType[3, float]
    b: TensorType[3, float]

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            args[k] = v.to(device)

        return Tube(**args)


@typechecked
@dataclass
class CollatedLineSegment:
    a: TensorType["N", 3, float]
    b: TensorType["N", 3, float]

    def to_device(self, device: torch.device):
        args = asdict(self)
        for k, v in args.items():
            args[k] = v.to(device)

        return CollatedLineSegment(**args)


def collates_line_segments(lines: List[LineSegment]) -> CollatedLineSegment:
    a = torch.cat([l.a for l in lines]).reshape(-1, 3)
    b = torch.cat([l.b for l in lines]).reshape(-1, 3)

    return CollatedLineSegment(a, b)
