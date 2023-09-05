import hydra
import taichi as ti
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

from smart_tree.data_types.cloud import Cloud, LabelledCloud
from smart_tree.o3d_abstractions.visualizer import o3d_viewer, ViewerItem


@hydra.main(
    version_base=None,
    config_path="../../smart_tree/conf/",
    config_name="pipeline.yaml",
)
def main(cfg: DictConfig):
    pipeline = instantiate(cfg.pipeline)

    clds = []

    for input_feats, coords, mask, filename in pipeline.model_inference.data_loader:
        print(torch.sum(mask == 0))

        LabelledCloud(xyz=input_feats, loss_mask=mask).view()

    # print(feats)
    # cld.view()


if __name__ == "__main__":
    main()
