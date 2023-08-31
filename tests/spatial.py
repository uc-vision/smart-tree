import numpy as np
from rtree import index
from tqdm import tqdm

# Generate some random ellipsoids and points for demonstration
# Replace this with your actual ellipsoid and point data
num_ellipsoids = 50000
num_points = 50000

# Example ellipsoid data (replace with your ellipsoid data)
ellipsoid_centers = np.random.rand(num_ellipsoids, 3)  # Replace with your ellipsoid centers
ellipsoid_axes = np.random.rand(num_ellipsoids, 3)  # Replace with your ellipsoid axes

# Example point data (replace with your point data)
points = np.random.rand(num_points, 3)  # Replace with your point coordinates

# Create an R-tree index for ellipsoids
p = index.Property()

p.dimension = 3

p.dat_extension = "data"

p.idx_extension = "index"
ellipsoid_index = index.Index("3d_index", properties=p)

# Insert ellipsoids into the R-tree index
for i in tqdm(range(len(ellipsoid_centers)), desc="Inserting ellipsoids"):
    ellipsoid = (
        i,
        [
            float(ellipsoid_centers[i][0] - ellipsoid_axes[i][0]),
            float(ellipsoid_centers[i][1] - ellipsoid_axes[i][1]),
            float(ellipsoid_centers[i][2] - ellipsoid_axes[i][2]),
            float(ellipsoid_centers[i][0] + ellipsoid_axes[i][0]),
            float(ellipsoid_centers[i][1] + ellipsoid_axes[i][1]),
            float(ellipsoid_centers[i][2] + ellipsoid_axes[i][2]),
        ],
    )
    ellipsoid_index.insert(ellipsoid[0], ellipsoid[1])
# Perform point-in-ellipse queries
points_inside_ellipsoids = {}  # Dictionary to store points inside each ellipsoid

for point_id, point in tqdm(enumerate(points)):
    ellipsoid_ids = list(ellipsoid_index.intersection(point))

    # # Check if the point is inside each ellipsoid
    # for ellipsoid_id in ellipsoid_ids:
    #     ellipsoid_center = ellipsoid_centers[ellipsoid_id]
    #     ellipsoid_axes = ellipsoid_axes[ellipsoid_id]

    #     # Check if the point is inside the ellipsoid using the equation of an ellipsoid
    #     is_inside = ((point[0] - ellipsoid_center[0]) / ellipsoid_axes[0]) ** 2 + (
    #         (point[1] - ellipsoid_center[1]) / ellipsoid_axes[1]
    #     ) ** 2 + ((point[2] - ellipsoid_center[2]) / ellipsoid_axes[2]) ** 2 <= 1

    #     if is_inside:
    #         if ellipsoid_id not in points_inside_ellipsoids:
    #             points_inside_ellipsoids[ellipsoid_id] = []
    #         points_inside_ellipsoids[ellipsoid_id].append(point_id)

# Now, points_inside_ellipsoids contains the mapping of ellipsoid IDs to the points inside them
