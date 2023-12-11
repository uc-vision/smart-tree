import logging
import time

import hydra
import taichi as ti

# from open3d_vis import render
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="../conf/vines",
    config_name="view-data-loader",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ti.init()
    log = logging.getLogger(__name__)

    torch.multiprocessing.set_start_method("spawn")

    loader = instantiate(cfg.capture_data_loader)
    log.info(f"Train Dataset Size: {len(loader.dataset)}")

    for i in range(10):
        start_time = time.time()
        for cloud_list in tqdm(loader):
            for c in cloud_list:
                c.view()

        print(f"Time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
