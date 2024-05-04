import torch
from spconv.pytorch.utils import gather_features_by_pc_voxel_id
from torch.nn import Module
from torch.utils.data import DataLoader

from smart_tree.data_types.cloud import Cloud, LabelledCloud, merge_clouds
from smart_tree.dataset.dataset import SingleTreeInference

from .sparse import split_sparse_features


def label_cloud(cloud: Cloud, voxel_preds, point_ids, mask):

    pt_radius = gather_features_by_pc_voxel_id(voxel_preds["radius"], point_ids)
    pt_direction = gather_features_by_pc_voxel_id(voxel_preds["direction"], point_ids)
    pt_class = gather_features_by_pc_voxel_id(voxel_preds["class_l"], point_ids)
    pt_mask = gather_features_by_pc_voxel_id(mask, point_ids).bool()

    class_l = torch.argmax(pt_class, dim=1, keepdim=True)
    medial_vector = torch.exp(pt_radius) * pt_direction

    cld = LabelledCloud(
        xyz=cloud.xyz,
        rgb=cloud.rgb,
        medial_vector=medial_vector,
        class_l=class_l,
    )

    return cld.filter(pt_mask)


class ModelInference:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,  # partial
        dataset: SingleTreeInference,  # partial
        device=torch.device("cuda:0"),
    ):
        self.model = model.to(device)
        self.data_loader = dataloader
        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def forward(self, cloud: Cloud | LabelledCloud):

        output_clouds = []

        data_loader = self.data_loader(self.dataset(cloud))

        for sparse_tensor, point_ids, input_clouds, voxel_masks in data_loader:

            sparse_tensor = sparse_tensor.to(self.device)
            preds = self.model.forward(sparse_tensor)

            voxel_features = split_sparse_features(sparse_tensor)

            cloud_preds = preds.split(
                [feats.shape[0] for feats in voxel_features],
                dim=0,
            )

            for preds, _ids, cld, mask in zip(
                cloud_preds,
                point_ids,
                input_clouds,
                voxel_masks,
            ):

                output_clouds.append(
                    label_cloud(cld, preds, _ids, mask).translate(-cld.offset_vector)
                )

        return merge_clouds(output_clouds).cpu()

    @staticmethod
    def from_cfg(cfg):
        return ModelInference(
            model_path=cfg.model_path,
            weights_path=cfg.weights_path,
            voxel_size=cfg.voxel_size,
            block_size=cfg.block_size,
            buffer_size=cfg.buffer_size,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
        )
