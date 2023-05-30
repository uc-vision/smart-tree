""" Script to sample meshes into point clouds.

  usage: 

  python smart_tree/scripts/sample-meshes.py --read_directory=/local/Datasets/Thingi10K/raw_meshes/ --save_directory=/local/Datasets/Thingi10K/point_clouds/ --scale=0.001 --voxel_size=0.01


"""

import json
import os
import random
from pathlib import Path

import click
import numpy as np

from tqdm import tqdm

from smart_tree.util.file import load_o3d_mesh, load_data_npz
from smart_tree.util.visualizer.view import o3d_viewer


@click.command()
@click.option(
    "--read_directory",
    type=click.Path(exists=True),
    prompt="read directory?",
)
@click.option(
    "--save_directory",
    type=click.Path(exists=True),
    prompt="save directory?",
)
@click.option(
    "--scale",
    type=float,
    prompt="scale?",
)
@click.option(
    "--voxel_size",
    type=float,
    prompt="voxel_size?",
)
def main(read_directory, save_directory, scale, voxel_size):
    files_paths = [str(path) for path in Path(read_directory).glob("*.stl")]

    pcd, skeleton = load_data_npz("/local/synthetic-trees/dataset/apple/apple_1.npz")

    for file_path in tqdm(files_paths):
        try:
            mesh = load_o3d_mesh(file_path)
            mesh = (
                mesh.scale(scale, mesh.get_center())
                .translate(-mesh.get_center())
                .paint_uniform_color(np.random.rand(3))
                .scale(100, mesh.get_center())
                .sample_points_uniformly(
                    int(mesh.get_surface_area() / (voxel_size**2))
                )
            )
        except:
            pass

        # o3d_viewer([mesh, pcd.to_o3d_cld()])


if __name__ == "__main__":
    main()
