
import hydra
import taichi as ti
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../../smart_tree/conf/",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    ti.init(arch=ti.gpu)

    loader = instantiate(cfg.train_data_loader)

    for cld in loader:
        print(cld)
        # cld.view()


if __name__ == "__main__":
    main()
