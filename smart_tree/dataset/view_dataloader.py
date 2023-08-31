import logging

import hydra
import taichi as ti
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from smart_tree.data_types.cloud import Cloud
from smart_tree.model.helper import get_batch
from smart_tree.model.sparse import split_sparse


def view(loader, cfg):
    for sp_input, targets, mask, filenames in tqdm(get_batch(loader, torch.device("cuda"), cfg.fp16)):
        split = split_sparse(sp_input)

        clds = []

        for coords, sp_feats in split:
            # clds.append(Cloud(sp_feats.reshape(-1, 3)).to_o3d_cld())
            # clds.append(Cloud(sp_feats.reshape(-1, 3)).to_o3d_cld())
            print(len(Cloud(sp_feats.reshape(-1, 3))))
        # o3d_viewer(clds)


@hydra.main(
    version_base=None,
    config_path="../conf/vines",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    ti.init(arch=ti.gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader, drop_last=False)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    view(test_loader, cfg)

    view(train_loader, cfg)
    view(val_loader, cfg)


if __name__ == "__main__":
    main()
