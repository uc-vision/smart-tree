import logging
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

import taichi as ti

@hydra.main(
    version_base=None,
    config_path="/local/smart-tree/smart_tree/conf/apple",
    config_name="train",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)
    
    ti.init(arch=ti.gpu,log_level=ti.INFO)
    torch.multiprocessing.set_start_method("spawn")

    train_loader = instantiate(cfg.train_data_loader)
    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")

    while True:
        start = time.time()

        for x in tqdm(train_loader):
            pass

        print(time.time() - start)


if __name__ == "__main__":
    main()
