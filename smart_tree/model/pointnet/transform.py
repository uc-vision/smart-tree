import torch


def split_cloud(cloud):
    return torch.tensor(cloud.xyz, device=cloud.device), torch.tensor(
        cloud.medial_vector, device=cloud.device
    )
