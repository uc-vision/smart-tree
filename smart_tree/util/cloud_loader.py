from dataclasses import fields
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from ..data_types.cloud import Cloud, LabelledCloud


class CloudLoader:
    def load(self, file: str | Path):
        file_path = Path(file)
        if file_path.suffix == ".npz":
            return self.load_numpy(file_path)
        else:
            return self.load_o3d(file_path)

    def load_o3d(self, file: str | Path) -> Cloud:
        try:
            pcd = o3d.io.read_point_cloud(str(file))
            return self._load_as_cloud(
                {"xyz": np.asarray(pcd.points), "rgb": np.asarray(pcd.colors)}, file
            )
        except Exception as e:
            raise ValueError(f"Error loading file {file}: {e}")

    def load_numpy(self, file_path: str | Path) -> Cloud | LabelledCloud:
        data = np.load(file_path, allow_pickle=True)
        return self._load(data, file_path)

    def _load(self, data, file_path: Path):
        optional_params = [
            f.name for f in fields(LabelledCloud) if f.default is not None
        ]

        for param in optional_params:
            if param in data:
                return self._load_as_labelled_cloud(data, file_path)

        return self._load_as_cloud(data, file_path)

    def _load_as_cloud(self, data, file_path: Path) -> Cloud:
        cloud_fields = self._extract_fields(data, Cloud, file_path)
        return Cloud(**cloud_fields)

    def _load_as_labelled_cloud(self, data, file_path: Path) -> LabelledCloud:
        labelled_cloud_fields = self._extract_fields(data, LabelledCloud, file_path)
        return LabelledCloud(**labelled_cloud_fields)

    def _extract_fields(self, data, cloud_type, file_path: Path) -> dict:
        fields_dict = {
            f.name: data[f.name] for f in fields(cloud_type) if f.name in data
        }
        for key, value in tqdm(fields_dict.items(), desc="Loading cloud", leave=False):
            if isinstance(value, np.ndarray):
                dtype = torch.long if key in ["class_l", "branch_ids"] else torch.float
                if value.ndim == 1:
                    fields_dict[key] = torch.from_numpy(value).type(dtype).unsqueeze(1)
                else:
                    fields_dict[key] = torch.from_numpy(value).type(dtype)

        fields_dict["filename"] = file_path

        if "vector" in data.keys():

            fields_dict["medial_vector"] = torch.from_numpy(data["vector"]).type(
                torch.float
            )

        return fields_dict