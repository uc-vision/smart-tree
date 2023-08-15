import os

import hydra
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .model.model_inference import ModelInference
from .skeleton.skeletonize import Skeletonizer
from .pipeline import Pipeline


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="tree-dataset",
)
def main(cfg: DictConfig):
    inferer = ModelInference.from_cfg(cfg.model_inference)
    skeletonizer = Skeletonizer.from_cfg(cfg.skeletonizer)
    pipeline = Pipeline.from_cfg(inferer, skeletonizer, cfg.pipeline)

    if "path" in dict(cfg):
        pipeline.process_cloud(Path(cfg.path))

    elif "directory" in dict(cfg):
        for p in os.listdir(cfg.directory):
            pipeline.process_cloud(Path(f"{cfg.directory}/{p}"))

    else:
        print("Please supply a path or directory to point clouds.")


if __name__ == "__main__":
    main()
