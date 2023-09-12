from pathlib import Path

import torch

from .data_types.cloud import Cloud, CloudLoader, LabelledCloud
from .data_types.tree import DisjointTreeSkeleton
from .dataset.augmentations import AugmentationPipeline
from .model.model_inference import ModelInference
from .o3d_abstractions.visualizer import o3d_viewer
from .skeleton.skeletonize import Skeletonizer
from .util.file import save_o3d_cloud, save_o3d_lineset, save_o3d_mesh


class Pipeline:
    def __init__(
        self,
        preprocessing: AugmentationPipeline,
        model_inference: ModelInference,
        skeletonizer: Skeletonizer,
        view_model_output=False,
        view_skeletons=False,
        save_outputs=False,
        save_path="/",
        branch_classes=[0],
        cmap=[[1, 0, 0], [0, 1, 0]],
        device=torch.device("cuda:0"),
    ):
        self.preprocessing = preprocessing
        self.model_inference = model_inference
        self.skeletonizer = skeletonizer

        self.view_model_output = view_model_output
        self.view_skeletons = view_skeletons

        self.cmap = torch.tensor(cmap, device=device)
        self.save_outputs = save_outputs
        self.save_path = save_path

        self.branch_classes = torch.tensor(branch_classes, device=device)
        self.device = device

    def run(self, path: Path):
        # Load point cloud and do any required preprocessing
        cloud: Cloud = CloudLoader().load(path).to_device(self.device)
        cloud = self.preprocessing(cloud)

        # Run cloud through network
        cloud: LabelledCloud = self.model_inference.forward(cloud)
        if self.view_model_output:
            cloud.view()

        # Filter only the branch points for skeletonizaiton
        branch_cloud: LabelledCloud = cloud.filter_by_class(self.branch_classes)

        # Run the branch cloud through skeletonization algorithm, then post process
        skeleton: DisjointTreeSkeleton = self.skeletonizer.forward(branch_cloud)

        # skeleton.to_pickle(
        #    "/mnt/harry/PhD/smart-tree/data/pickled_unconnected_skeletons/apple_10.pkl"
        # )

        # View skeletonization results
        if self.view_skeletons:
            o3d_viewer(skeleton.viewer_items() + cloud.viewer_items(), line_width=5)

        if self.save_outputs:
            sp = self.save_path
            save_o3d_lineset(f"{sp}/skeleton.ply", skeleton.as_o3d_lineset())
            save_o3d_mesh(f"{sp}/mesh.ply", skeleton.as_o3d_tube())
            save_o3d_cloud(f"{sp}/cloud.ply", cloud.as_o3d_cld())
            save_o3d_cloud(f"{sp}/seg_cld.ply", cloud.as_o3d_segmented_cld(self.cmap))
