from typing import List

import torch

from smart_tree.data_types.cloud import Cloud, LabelledCloud


def model_output_to_labelled_clds(
    model_input,
    preds,
    metadata,
    cmap,
) -> List[Cloud]:
    clouds = []

    num_clouds = (model_input.indices[:, 0]).max().int() + 1

    for i in range(num_clouds):
        mask = i == model_input.indices[:, 0]

        xyz = model_input.features[:, :3][mask]
        predicted_class = torch.argmax(preds["class_l"][mask], dim=1).int()
        rgb = torch.tensor(cmap, device=preds["class_l"].device)[predicted_class]

        lc = LabelledCloud(
            xyz,
            rgb=rgb,
            class_l=predicted_class.unsqueeze(1),
            filename=metadata.filename[i],
        )

        clouds.append(lc)

    return clouds


def identity_collate_fn(batch):
    return batch

    # return to_labelled_clds(
    #     sparse_input.indices[:, 0],
    #     sparse_input.features[:, :3],
    #     sparse_input.features[:, 3:6],
    #     model_output,
    #     cmap,
    #     filenames,
    # )


# def split_outputs(features, mask):
#     radii = torch.exp(features["radius"][mask])
#     direction = features["medial_direction"][mask]
#     class_l = torch.argmax(features["class_l"], dim=1)[mask]

#     return radii, direction, class_l


# def to_labelled_clds(
#     cloud_ids,
#     coords,
#     rgb,
#     model_output,
#     cmap,
#     filenames,
# ) -> List[Cloud]:
#     num_clouds = cloud_ids.max() + 1
#     clouds = []

#     # assert rgb.shape[1] > 0

#     for i in range(num_clouds):
#         mask = cloud_ids == i
#         xyz = coords[mask]
#         rgb = torch.rand(xyz.shape)  # rgb[mask]

#         radii, direction, class_l = split_outputs(model_output, mask)

#         labelled_cloud = Cloud(
#             xyz=xyz,
#             rgb=rgb,
#             medial_vector=radii * direction,
#             class_l=class_l,
#             filename=Path(filenames[i]),
#         )

#         clouds.append(labelled_cloud.to_device(torch.device("cpu")))

#     return clouds
