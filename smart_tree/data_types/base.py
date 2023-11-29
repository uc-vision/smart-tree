from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
@dataclass
class Base:
    def apply_to_tensors(self, func: Callable[[Any], Any]):
        return {
            k: (func(v) if isinstance(v, torch.Tensor) else v)
            for k, v in asdict(self).items()
        }

    def apply_to_2d_tensors(self, func: Callable[[Any], Any]):
        return {
            k: (func(v) if isinstance(v, torch.Tensor) and v.dim() == 2 else v)
            for k, v in asdict(self).items()
        }

    def filter(self, mask: TensorType["N", torch.bool]):
        filtered_args = self.apply_to_2d_tensors(lambda v: v[mask])
        return self.__class__(**filtered_args)

    def to_device(self, device: torch.device):
        device_args = self.apply_to_tensors(lambda v: v.to(device))
        return self.__class__(**device_args)

    def pin_memory(self):
        pinned_args = self.apply_to_tensors(lambda v: v.pin_memory())
        return self.__class__(**pinned_args)
