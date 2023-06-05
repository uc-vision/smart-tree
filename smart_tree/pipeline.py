from pathlib import Path

import numpy as np
import torch
import time

from .data_types.cloud import LabelledCloud, Cloud
from .data_types.tree import TreeSkeleton, DisjointTreeSkeleton
from hydra.utils import instantiate

from .util.visualizer.view import o3d_viewer
from .util.misc import to_numpy

from .util.file import (
    load_adtree_skeleton,
    load_cloud,
    save_o3d_cloud,
    save_o3d_lineset,
    save_o3d_mesh,
    o3d_cloud,
)

from .util.mesh.geometries import o3d_lines_between_clouds
from .dataset.augmentations import AugmentationPipeline


class Pipeline:
    def __init__(
        self,
        inferer,
        skeletonizer,
        preprocessing_cfg,
        repair_skeletons=False,
        smooth_skeletons=False,
        smooth_kernel_size=0,
        prune_skeletons=False,
        min_skeleton_radius=0.0,
        min_skeleton_length=1000,
        view_model_output=False,
        view_skeletons=False,
        save_outputs=False,
        branch_classes=[0],
        cmap=[[1, 0, 0], [0, 1, 0]],
        device=torch.device("cuda:0"),
    ):
        self.inferer = inferer
        self.skeletonizer = skeletonizer

        self.preprocessing = AugmentationPipeline.from_cfg(
            instantiate(preprocessing_cfg)
        )

        self.repair_skeletons = repair_skeletons
        self.smooth_skeletons = smooth_skeletons
        self.smooth_kernel_size = smooth_kernel_size
        self.prune_skeletons = prune_skeletons

        self.min_skeleton_radius = min_skeleton_radius
        self.min_skeleton_length = min_skeleton_length

        self.view_model_output = view_model_output
        self.view_skeletons = view_skeletons

        self.save_outputs = save_outputs

        self.branch_classes = branch_classes
        self.cmap = np.asarray(cmap)
        self.device = device

    def process_cloud(self, path: Path):
        # Load point cloud
        cloud: Cloud = load_cloud(path).to_device(self.device)
        cloud = self.preprocessing(cloud)

        # Run point cloud through model to predict class, radius, direction
        lc: LabelledCloud = self.inferer.forward(cloud).to_device(self.device)
        if self.view_model_output:
            lc.view(self.cmap)

        # Filter only the branch points for skeletonizaiton
        branch_cloud: LabelledCloud = lc.filter_by_class(self.branch_classes)

        # Run the branch cloud through skeletonization algorithm, then post process
        skeleton: DisjointTreeSkeleton = self.skeletonizer.forward(branch_cloud)

        self.post_process(skeleton)

        # View skeletonization results
        if self.view_skeletons:
            o3d_viewer(
                [
                    skeleton.to_o3d_tube(),
                    skeleton.to_o3d_lineset(),
                    skeleton.to_o3d_tube(colour=False),
                    cloud.to_o3d_cld(),
                ],
                line_width=5,
            )

        if self.save_outputs:
            save_o3d_lineset("skeleton.ply", skeleton.to_o3d_lineset())
            save_o3d_mesh("mesh.ply", skeleton.to_o3d_tube())
            save_o3d_cloud("cloud.ply", cloud.to_o3d_cld())

    def post_process(self, skeleton: DisjointTreeSkeleton):
        if self.prune_skeletons:
            skeleton.prune(
                min_length=self.min_skeleton_length,
                min_radius=self.min_skeleton_radius,
            )

        if self.repair_skeletons:
            skeleton.repair()

        if self.smooth_skeletons:
            skeleton.smooth(self.smooth_kernel_size)

    @staticmethod
    def from_cfg(inferer, skeletonizer, cfg):
        return Pipeline(
            inferer,
            skeletonizer,
            preprocessing_cfg=cfg.preprocessing,
            repair_skeletons=cfg.repair_skeletons,
            smooth_skeletons=cfg.smooth_skeletons,
            smooth_kernel_size=cfg.self.smooth_kernel_size,
            prune_skeletons=cfg.prune_skeletons,
            min_skeleton_radius=cfg.min_skeleton_radius,
            min_skeleton_length=cfg.min_skeleton_length,
            view_model_output=cfg.view_model_output,
            view_skeletons=cfg.view_skeletons,
            save_outputs=cfg.save_outputs,
            branch_classes=cfg.branch_classes,
            cmap=cfg.cmap,
        )
