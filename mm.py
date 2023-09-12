import taichi as ti

ti.init(arch=ti.cpu)  # Initialize Taichi (or ti.gpu for GPU)

# Define the size of the "matrix"
N = 5

# Define separate fields for each component
x_component = ti.field(dtype=ti.f32, shape=N)
y_component = ti.field(dtype=ti.f32, shape=N)
z_component = ti.field(dtype=ti.f32, shape=N)

# Function to set a "row" in the matrix
@ti.kernel
def set_row(row_idx: ti.i32, x_val: ti.f32, y_val: ti.f32, z_val: ti.f32):
    x_component[row_idx] = x_val
    y_component[row_idx] = y_val
    z_component[row_idx] = z_val

# Function to get a "row" from the matrix
@ti.kernel
def get_row(row_idx: ti.i32) -> ti.Vector:
    return ti.Vector([x_component[row_idx], y_component[row_idx], z_component[row_idx]])

# Assign a vector to a "row"
set_row(2, 1.0, 2.0, 3.0)

# Access the assigned "row"
assigned_vector = get_row(2)

print("Assigned Vector:", assigned_vector.to_numpy())
