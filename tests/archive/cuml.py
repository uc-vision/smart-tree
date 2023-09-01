import cupy as cp
from cuml.neighbors import NearestNeighbors

# Define the number of points
N = 500000
M = 500000

# Generate some random Nx3 and Mx3 points on the GPU
points1 = cp.random.rand(N, 3)
points2 = cp.random.rand(M, 3)

# Define the number of nearest neighbors to find
k = 1

# Initialize the NearestNeighbors model
nn = NearestNeighbors(n_neighbors=k, metric="euclidean")

# Fit the model to points2 (build KD-tree on the GPU)
nn.fit(points2)

# Perform the nearest neighbor search for points1 on the GPU
distances, indices = nn.kneighbors(points1)

# 'distances' contains the distances from each point in points1 to its nearest neighbor in points2
# 'indices' contains the indices of the nearest neighbors in points2

# Print the results
print("Distances:", distances)
print("Indices:", indices)
