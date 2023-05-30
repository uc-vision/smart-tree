import torch

from smart_tree.data_types.cloud import Cloud
from smart_tree.util.file import load_yaml
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import hydra

import time


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="test-dataloader",
)
def main(cfg: DictConfig):
    dataset = instantiate(cfg.dataset, mode="train")
    train_dataloader = instantiate(cfg.data_loader, dataset)

    start_time = time.time()
    for data in tqdm(train_dataloader):
        pass

        # times = [
        #     s.elapsed_time(e) for s, e in zip(dataset.start_events, dataset.end_events)
        # ]

        # print(times)


if __name__ == "__main__":
    main()
