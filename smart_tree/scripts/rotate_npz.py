import argparse
from pathlib import Path

import torch

from smart_tree.util.file import (
    CloudLoader,
    save_data_npz,
    load_cloud_and_skeleton,
)
from smart_tree.util.maths import euler_angles_to_rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument(
        "file_path",
        help="File Path of tree.npz",
        default=None,
        type=Path,
    )

    parser.add_argument(
        "-lw",
        "--line_width",
        help="Width of visualizer lines",
        default=1,
        type=int,
    )
    return parser.parse_args()


def paths_from_args(args, glob="*.npz"):
    if not args.file_path.exists():
        raise ValueError(f"File {args.file_path} does not exist")

    if args.file_path.is_file():
        print(f"Loading data from file: {args.file_path}")
        return [args.file_path]

    if args.file_path.is_dir():
        print(f"Loading data from directory: {args.file_path}")
        files = args.file_path.glob(glob)
        if files == []:
            raise ValueError(f"No npz files found in {args.file_path}")
        return files


def load_data(paths):
    for path in paths:
        yield load_cloud_and_skeleton(path)


def main():
    args = parse_args()

    # loader = CloudLoader()

    rotation_mat = euler_angles_to_rotation(torch.tensor([1.57, 0.0, 0.0])).float()

    # rotated_clouds = [cloud.rotate(rotation_mat) for cloud in clouds]

    # translated_cloud = [cloud.translate(-cloud.centre) for cloud in rotated_clouds]

    for cloud, skeleton in load_data(paths_from_args(args)):
        print("Cloud Loaded...")

        print(cloud.medial_vector)

        cloud.view()

        cloud = cloud.rotate(rotation_mat)

        save_name = f"/local/smart-tree/data/vines/synthetic-vine-pcds-3-rotated/{cloud.filename.name}"

        save_data_npz(save_name, skeleton, cloud)

        cloud.view()


if __name__ == "__main__":
    main()
