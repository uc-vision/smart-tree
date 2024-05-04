import json
from pathlib import Path

import torch
import torch.utils.data

from tqdm import tqdm
from beartype.typing import Any, List, Literal, Optional, Union
from typeguard import typechecked
from ..data_types.cloud import Cloud
from ..model.voxelize import SparseVoxelizer
from ..util.cloud_loader import CloudLoader
from .augmentations import AugmentationPipeline
from ..util.maths import cube_filter

FilePath = Union[str, Path]
DatasetModes = Literal["train", "validation", "test", "capture"]
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
        loader=CloudLoader(),
    ):
        self.transform = transform
        self.augmentation = augmentation
        self.tree_paths = tree_paths
        self.device = torch.device(device)
        self.loader = loader
        self.cache = {} if cache else False

    def load(self, filename):  # -> Cloud | LabelledCloud:
        if self.cache is False:
            return self.loader.load(filename)

        if filename not in self.cache:
            self.cache[filename] = self.loader.load(filename).pin_memory()

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
        voxelizer: SparseVoxelizer,
        block_size: float = 4,
        buffer_size: float = 0.4,
        min_points=20,
    ):

        self.cloud = cloud
        self.voxelizer = voxelizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.min_points = min_points

        self.compute_blocks()

    def compute_blocks(self):

        xyz_quantized = torch.div(
            self.cloud.xyz,
            self.block_size,
            rounding_mode="floor",
        )
        block_ids, pnt_counts = torch.unique(
            xyz_quantized,
            return_counts=True,
            dim=0,
        )

        valid_block_ids = block_ids[pnt_counts > self.min_points]
        self.block_centres = (valid_block_ids * self.block_size) + (self.block_size / 2)

        self.clouds: List[Cloud] = []

        for centre in tqdm(self.block_centres, desc="Computing blocks"):
            mask = cube_filter(
                self.cloud.xyz,
                centre,
                self.block_size + (self.buffer_size * 2),
            )
            block_cloud = self.cloud.filter(mask)

            self.clouds.append(block_cloud)

    def __getitem__(self, idx):
        block_centre = self.block_centres[idx]
        block_cloud = self.clouds[idx]

        block_cloud.mask = cube_filter(
            block_cloud.xyz,
            block_centre,
            self.block_size,
        ).unsqueeze(1)

        offset_vector = -torch.min(block_cloud.xyz, dim=0)[0]
        block_cloud = block_cloud.offset(offset_vector)
        voxel_cloud = self.voxelizer(block_cloud)

        return voxel_cloud


    def __len__(self):
        return len(self.clouds)


def load_dataloader(
    cloud: Cloud,
    voxel_size: float,
    block_size: float,
    buffer_size: float,
    num_workers: float,
    batch_size: float,
):
    dataset = SingleTreeInference(cloud, voxel_size, block_size, buffer_size)

    return DataLoader(dataset, batch_size, num_workers)  # collate_fn=merge_clouds)
