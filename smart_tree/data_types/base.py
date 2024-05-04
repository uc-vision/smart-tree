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

    def filter(self, mask: TensorType["N"]):
        filtered_args = self.apply_to_2d_tensors(lambda v: v[mask])
        return self.__class__(**filtered_args)

    def to_device(self, device: torch.device):
        device_args = self.apply_to_tensors(
            lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
        )
        return self.__class__(**device_args)

    def pin_memory(self):
        pinned_args = self.apply_to_tensors(
            lambda v: v.pin_memory() if isinstance(v, torch.Tensor) else v
        )
        return self.__class__(**pinned_args)

    def print_tensors_with_nans(self):
        def has_nans(tensor):
            if torch.isnan(tensor).any():
                return True
            return False

        tensors_with_nans = self.apply_to_tensors(
            lambda v: has_nans(v) if isinstance(v, torch.Tensor) else None
        )
        for key, value in tensors_with_nans.items():
            if value is True:
                print(f"Tensor '{key}' contains NaNs.")

    def cpu(self):
        return self.to_device(torch.device("cpu"))

    def cuda(self):
        return self.to_device(torch.device("cuda"))
