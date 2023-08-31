import torch
from sklearn.neighbors import KDTree

# Define the number of points
N = 500000
M = 500000

# Define the batch size for splitting
batch_size = 30000

# Generate some random Nx3 and Mx3 points
points1 = torch.rand(N, 3)  # Nx3 tensor
points2 = torch.rand(M, 3)  # Mx3 tensor

# Initialize an empty tensor to store distances

# Build KD-tree for points2
points2_np = points2.numpy()
kdtree = KDTree(points2_np)

# Loop through mini-batches
for i in range(0, N, batch_size):
    batch_points1 = points1[i : i + batch_size]

    # Query the KD-tree for each batch of points
    distances_batch, _ = kdtree.query(batch_points1.numpy(), k=batch_size)

    # Store the results in the distances tensor

# 'distances' now contains the pairwise distances for all points
print(distances)
