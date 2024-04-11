import json
from pathlib import Path
from beartype.typing import Any, Literal, Optional, Union, List

import torch
import torch.utils.data
from typeguard import typechecked

from ..data_types.cloud import Cloud
from .augmentations import AugmentationPipeline

FilePath = Union[str, Path]
DatasetModes = Literal["train", "validation", "test", "capture", "unlabelled"]
Device = Literal["cpu", "cuda"]


@typechecked
class Dataset:
    def __init__(
        self,
        tree_paths: List[Path] | List[str],
        transform: Optional[callable] = None,
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = None,
        device: Device = "cpu",
    ):
        self.transform = transform
        self.augmentation = augmentation
        self.tree_paths = tree_paths
        self.device = torch.device(device)

        self.cache = {} if cache else False

    def load(self, filename):  # -> Cloud | LabelledCloud:
        if self.cache is False:
            return Cloud.from_file(filename)

        if filename not in self.cache:
            self.cache[filename] = Cloud.from_file(filename).pin_memory()

        return self.cache[filename]

    def __getitem__(self, idx):
        cld = self.load(self.tree_paths[idx])
        try:
            return self.process_cloud(cld)
        except Exception:
            print(f"Exception processing {cld}")
            raise

    def __len__(self):
        return len(self.tree_paths)

    def process_cloud(self, cld) -> Any:  #: Cloud | LabelledCloud

        cld = cld.to_device(self.device)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        if self.transform != None:
            cld = self.transform(cld)

        return cld

    @staticmethod
    def from_directory(directory: Union[str, Path], *args, **kwargs) -> "Dataset":
        assert Path(directory).is_dir(), f"Directory path is invalid: {directory}"
        valid_extensions = ["*.ply", "*.pcd", "*.npz"]
        tree_paths = [
            str(file)
            for ext in valid_extensions
            for file in Path(directory).rglob(ext)
            if file.is_file()
        ]
        if len(tree_paths) == 0:
            raise ValueError("Directory contains no valid data")

        return Dataset(tree_paths=tree_paths, *args, **kwargs)

    @staticmethod
    def from_path(path: FilePath, *args, **kwargs) -> "Dataset":
        assert Path(path).exists(), f"path is invalid: {path}"
        return Dataset(tree_paths=[path], *args, **kwargs)

    @staticmethod
    def from_json(
        json_path: FilePath,
        directory: FilePath,
        mode: DatasetModes,
        *args,
        **kwargs,
    ) -> "Dataset":

        json_data = json.load(open(json_path))
        paths = [Path(f"{directory}/{p}") for p in json_data[mode]]
        invalid_paths = [path for path in paths if not path.exists()]
        assert not invalid_paths, f"Missing {len(invalid_paths)} files: {invalid_paths}"
        return Dataset(tree_paths=paths, *args, **kwargs)

    @staticmethod
    def from_auto(input_path: FilePath, *args, **kwargs) -> "Dataset":
        path_obj = Path(input_path)

        if path_obj.is_dir():
            return Dataset.from_directory(directory=input_path, *args, **kwargs)
        elif path_obj.is_file():
            return Dataset.from_path(path=input_path, *args, **kwargs)
        else:
            raise ValueError(f"Input path is not a file nor a directory: {input_path}")


class SingleTreeInference:

    def __init__(
        self,
        cloud: Cloud,
        block_size: float = 4.0,
        buffer_size: float = 0.4,
        min_points: int = 20,
        augmentation: Optional[callable] = None,
        transform: Optional[callable] = None,
    ):
        self.cloud = cloud
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.augmentation = augmentation
        self.transform = transform

        xyz_quantized = torch.div(self.cloud.xyz, block_size, rounding_mode="floor")
        block_ids, pnt_counts = torch.unique(xyz_quantized, return_counts=True, dim=0)

        valid_block_ids = block_ids[pnt_counts > min_points]
        self.block_centres = (valid_block_ids * block_size) + (block_size / 2)

    def __getitem__(self, idx) -> Any:
        centre = self.block_centres[idx]

        xyzmin = centre - (self.block_size / 2)
        xyzmax = centre + (self.block_size / 2)

        block_max_xyz = xyzmax + self.buffer_size
        block_min_xyz = xyzmin - self.buffer_size

        block_mask = (self.cloud.xyz > block_min_xyz) & (self.cloud.xyz < block_max_xyz)
        block_mask = torch.all(block_mask, dim=1)

        block_cld = self.cloud.filter(block_mask)

        buffer_mask = (block_cld.xyz > xyzmin) & (block_cld.xyz < xyzmax)
        buffer_mask = torch.all(buffer_mask, dim=1).unsqueeze(1)

        block_cld.loss_mask = buffer_mask

        if self.augmentation:
            block_cld = self.augmentation(block_cld)

        if self.transform:
            block_cld = self.transform(block_cld)

        return block_cld

    def __len__(self):
        return len(self.block_centres)
