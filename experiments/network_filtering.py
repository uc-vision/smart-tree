import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from smart_tree.model.model_inference import ModelInference
from smart_tree.util.file import load_cloud

from smart_tree.skeleton.graph import relative_density

from smart_tree.o3d_abstractions.geometries import o3d_scalar_cloud
from smart_tree.o3d_abstractions.visualizer import o3d_viewer
from smart_tree.dataset.augmentations import CentreCloud


def view_density(cloud):

    density = relative_density(
        cloud.medial_pts.cuda(),
        cloud.radius.cuda(),
    )

    o3d_viewer([o3d_scalar_cloud(cloud.medial_pts, density.cpu().unsqueeze(1))])


def main():

    cloud = load_cloud(
        Path("/local/Downloads/Dataset_tree/2023-01-16_3/2023-01-16_3_000001.txt")
    )

    print(cloud.xyz.shape)

    infer = ModelInference(
        model_path=Path("smart_tree/model/weights/noble-elevator-58_model.pt"),
        weights_path=Path(
            "smart_tree/model/weights/noble-elevator-58_model_weights.pt"
        ),
        voxel_size=0.01,
        block_size=4,
        buffer_size=0.4,
        num_workers=8,
        batch_size=1,
    )

    aug = CentreCloud()

    cloud = aug(cloud)

    network_cloud = infer.forward(cloud)

    network_cloud.view()

    print(network_cloud.xyz.shape)

    view_density(network_cloud)


if __name__ == "__main__":
    main()
