import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from smart_tree.data_types.cloud import LabelledCloud


@hydra.main(
    version_base=None,
    config_path="../../smart_tree/conf/",
    config_name="pipeline.yaml",
)
def main(cfg: DictConfig):
    cfg.batch_size = 1
    pipeline = instantiate(cfg.pipeline)

    clds = []

    for input_feats, coords, mask, filename in pipeline.model_inference.data_loader:
        print(torch.sum(mask == 0))

        LabelledCloud(xyz=input_feats, loss_mask=mask).view()

    # print(feats)
    # cld.view()


if __name__ == "__main__":
    main()
