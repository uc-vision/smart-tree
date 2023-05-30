import glob
from pathlib import Path

import numpy as np


from tqdm import tqdm

from smart_tree.util.file import load_data_npz
from smart_tree.data_types.cloud import LabelledCloud


# " /local/uc-vision/smart-tree/data/new_npz/
# "/local/uc-vision/dataset/branches"
def main():
    for npz in tqdm(Path("/local/uc-vision/smart-tree/data/new_npz/").rglob("*.npz")):
        data = np.load(npz, allow_pickle=False, fix_imports=False, mmap_mode="r")
        # xyz = data["xyz"]
        cld = (data["xyz"], data["rgb"], data["vector"], data["class_l"])
        # cld = LabelledCloud.from_numpy(
        #     xyz=data["xyz"],
        #     rgb=data["rgb"],
        #     vector=data["vector"],
        #     class_l=data["class_l"],
        # )

        # np.savez(
        #     file=f"/local/uc-vision/smart-tree/data/new_npz/{npz.stem}",
        #     xyz=data["xyz"].astype(np.float32),
        #     rgb=data["rgb"].astype(np.float32),
        #     vector=data["vector"].astype(np.float32),
        #     class_l=data["class_l"].astype(np.float32),
        # )


if __name__ == "__main__":
    main()
