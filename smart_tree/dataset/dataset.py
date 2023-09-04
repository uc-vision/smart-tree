import json
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.utils.data
from spconv.pytorch.utils import PointToVoxel
from torch.utils.data import DataLoader
from tqdm import tqdm
from typeguard import typechecked

from ..data_types.cloud import Cloud, CloudLoader, LabelledCloud
from ..util.maths import cube_filter
from ..util.misc import at_least_2d
from .augmentations import AugmentationPipeline


@typechecked
class Dataset:
    def __init__(
        self,
        json_path: Path | str,
        directory: Path | str,
        mode: Literal["train", "validation", "test", "unlabelled"],
        transform: Optional[callable] = None,
        augmentation: Optional[AugmentationPipeline] = None,
        cache: bool = False,
        device=torch.device("cuda:0"),
    ):
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.device = device

        assert Path(json_path).is_file(), f"JSON path is invalid: '{json_path}'"
        assert Path(directory).is_dir(), "Directory path is invalid: {directory}"

        json_data = json.load(open(json_path))

        if self.mode == "train":
            tree_paths = json_data["train"]
        elif self.mode == "validation":
            tree_paths = json_data["validation"]
        elif mode == "test":
            tree_paths = json_data["test"]

        self.full_paths = [Path(f"{directory}/{p}") for p in tree_paths]
        invalid_paths = [path for path in self.full_paths if not path.exists()]
        assert not invalid_paths, f"Missing {len(invalid_paths)} files: {invalid_paths}"

        self.cache = {} if cache else None
        self.cloud_loader = CloudLoader()

    def load(self, filename) -> Cloud | LabelledCloud:
        if self.cache is None:
            return self.cloud_loader.load(filename)

        if filename not in self.cache:
            self.cache[filename] = self.cloud_loader.load(filename).pin_memory()

        return self.cache[filename]

    def __getitem__(self, idx):
        cld = self.load(self.full_paths[idx])
        try:
            return self.process_cloud(cld)
        except Exception:
            print(f"Exception processing {cld}")
            raise

    def __len__(self):
        return len(self.full_paths)

    def process_cloud(self, cld: Cloud | LabelledCloud) -> List[Cloud | LabelledCloud]:
        cld = cld.to_device(self.device)

        if self.augmentation != None:
            cld = self.augmentation(cld)

        if self.transform != None:
            cld = self.transform(cld)

        return cld


class SingleTreeInference:
    def __init__(
        self,
        cloud: Cloud,
        device=torch.device("cuda:0"),
        block_size: float = 1.0,
        buffer_size: float = 0.4,
        min_points: int = 20,
    ):
        self.cloud = cloud

        xyz_quantized = torch.div(cloud.xyz, block_size, rounding_mode="floor")
        block_ids, pnt_counts = torch.unique(xyz_quantized, return_counts=True, dim=0)

        valid_block_ids = block_ids[pnt_counts > min_points]
        self.block_centres = (valid_block_ids * block_size) + (block_size / 2)

    def __getitem__(self, idx):
        block_centre = self.block_centres[idx]
        cloud: Cloud = self.clouds[idx]

        xyzmin, _ = torch.min(cloud.xyz, axis=0)
        xyzmax, _ = torch.max(cloud.xyz, axis=0)

        surface_voxel_generator = PointToVoxel(
            vsize_xyz=[self.voxel_size] * 3,
            coors_range_xyz=[
                xyzmin[0],
                xyzmin[1],
                xyzmin[2],
                xyzmax[0],
                xyzmax[1],
                xyzmax[2],
            ],
            num_point_features=6,
            max_num_voxels=len(cloud),
            max_num_points_per_voxel=1,
        )

        feats, coords, _, voxel_id_tv = surface_voxel_generator.generate_voxel_with_id(
            torch.cat((cloud.xyz, cloud.rgb), dim=1).contiguous()
        )  #

        indice = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
        coords = torch.cat((indice, coords), dim=1)

        feats = feats.squeeze(1)
        coords = coords.squeeze(1)

        mask = cube_filter(feats[:, :3], block_centre, self.block_size)

        return feats, coords, mask, self.file_name

    def __len__(self):
        return len(self.clouds)

    # def compute_blocks(self):
    #     self.xyz_quantized = torch.div(
    #         self.cloud.xyz, self.block_size, rounding_mode="floor"
    #     )
    #     self.block_ids, pnt_counts = torch.unique(
    #         self.xyz_quantized, return_counts=True, dim=0
    #     )

    #     # Remove blocks that have less than specified amount of points...
    #     self.block_ids = self.block_ids[pnt_counts > self.min_points]
    #     self.block_centres = (self.block_ids * self.block_size) + (self.block_size / 2)

    #     self.clouds: List[Cloud] = []

    #     for centre in tqdm(self.block_centres, desc="Computing blocks...", leave=False):
    #         mask = cube_filter(
    #             self.cloud.xyz,
    #             centre,
    #             self.block_size + (self.buffer_size * 2),
    #         )
    #         block_cloud = self.cloud.filter(mask).to_device(torch.device("cpu"))

    #         self.clouds.append(block_cloud)

    #     self.block_centres = self.block_centres.to(torch.device("cpu"))


def load_dataloader(
    cloud: Cloud,
    voxel_size: float,
    block_size: float,
    buffer_size: float,
    num_workers: float,
    batch_size: float,
):
    dataset = SingleTreeInference(cloud, voxel_size, block_size, buffer_size)

    return DataLoader(dataset, batch_size, num_workers, collate_fn=batch_collate)
