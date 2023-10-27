import os
from pathlib import Path
import yaml
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra import initialize, compose

from .pipeline import Pipeline

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="pipeline",
)
def main(cfg: DictConfig):
    pipeline = instantiate(cfg.pipeline)

    if "path" in dict(cfg):
        pipeline.run(Path(cfg.path))
    elif "directory" in dict(cfg):
        for p in os.listdir(cfg.directory):
            pipeline.run(Path(f"{cfg.directory}/{p}"))
    else:
        print("Please supply a path or directory to point clouds.")


if __name__ == "__main__":
    main()
