import argparse
from pathlib import Path

import torch

from smart_tree.data_types.cloud import CloudLoader
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


def main():
    args = parse_args()

    loader = CloudLoader()
    clouds = [loader.load(filename) for filename in paths_from_args(args)]

    rotation_mat = euler_angles_to_rotation(torch.tensor([1.57, 0.0, 0.0])).float()

    # rotated_clouds = [cloud.rotate(rotation_mat) for cloud in clouds]

    # translated_cloud = [cloud.translate(-cloud.centre) for cloud in rotated_clouds]

    for cloud in clouds:
        cloud = cloud.rotate(rotation_mat)

        # save_name = cloud.filename.name
        # save_cloud(
        #     cloud,
        #     f"/mnt/harry/PhD/smart-tree/data/synthetic-vine-pcds-3-rotated/{save_name}",
        # )
        cloud.view()


if __name__ == "__main__":
    main()
