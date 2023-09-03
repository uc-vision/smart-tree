import logging

import hydra
import taichi as ti
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def collate(batch):
    return batch[0]


@hydra.main(
    version_base=None,
    config_path="../../smart_tree/conf/",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    ti.init(arch=ti.gpu)

    loader = instantiate(cfg.train_data_loader)

    for cld in loader:
        cld.view()


if __name__ == "__main__":
    main()
