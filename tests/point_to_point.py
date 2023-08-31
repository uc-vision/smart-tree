import torch

# Define the number of points
N = 500000
M = 500000

# Define the batch size for splitting
batch_size = 2000

# Generate some random Nx3 and Mx3 points
points1 = torch.rand(N, 3)  # Nx3 tensor
points2 = torch.rand(M, 3).cuda()  # Mx3 tensor


# Loop through mini-batches
for i in range(0, N, batch_size):
    batch_points1 = points1[i : i + batch_size]

    # Compute pairwise distances for the current mini-batch
    batch_distances = torch.norm(batch_points1[:, None, :].cuda() - points2[None, :, :], dim=2)
