from typing import List

import numpy as np
import open3d as o3d
import torch

from ..util.maths import gen_tangents, random_unit, vertex_dirs
from ..util.misc import as_numpy


@as_numpy
def o3d_mesh(verts, tris):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    return mesh.compute_triangle_normals()


def o3d_merge_meshes(
    meshes: List[o3d.geometry.TriangleMesh],
    colour=None,
    random_colours=None,
):
    if len(meshes) < 1:
        raise ValueError("Meshes List is Empty")

    if random_colours:
        for m in meshes:
            m.paint_uniform_color(torch.rand(3))

    mesh = meshes[0]
    for m in meshes[1:]:
        mesh += m

    if colour != None:
        return mesh.paint_uniform_color(colour)

    return mesh


def o3d_merge_linesets(line_sets, colour=[0, 0, 0]):
    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate([ls.lines + offset for ls, offset in zip(line_sets, offsets)])

    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(idxs)
    ).paint_uniform_color(colour)


@as_numpy
def o3d_sphere(xyz, radius, colour=(1, 0, 0)):
    return (
        o3d.geometry.TriangleMesh.create_sphere(radius)
        .translate(np.asarray(xyz))
        .paint_uniform_color(colour)
    )


@as_numpy
def o3d_spheres(xyzs, radii, colour=None, colours=None):
    spheres = [o3d_sphere(xyz, r) for xyz, r in zip(xyzs, radii)]
    return paint_o3d_geoms(spheres, colour, colours)


@as_numpy
def o3d_line_set(vertices, edges, colour=None, colours=None):
    line_set = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges)
    )
    if colour is not None:
        line_set = line_set.paint_uniform_color(colour)

    if colours is not None:
        line_set.colors = o3d.utility.Vector3dVector(colours * 255)

    return line_set


@as_numpy
def o3d_line_sets(vertices, edges):
    return [o3d_line_set(v, e) for v, e in zip(vertices, edges)]


@as_numpy
def o3d_path(vertices, colour=(1, 0, 0)):
    idx = torch.arange(vertices.shape[0] - 1)
    return o3d_line_set(
        vertices, torch.column_stack((idx, idx + 1))
    ).paint_uniform_color(colour)


@as_numpy
def o3d_paths(vertices):
    return [o3d_path(v) for v in vertices]


@as_numpy
def o3d_merge_clouds(points_clds):
    points = np.concatenate([np.asarray(pcd.points) for pcd in points_clds])
    colors = np.concatenate([np.asarray(pcd.colors) for pcd in points_clds])

    return o3d_cloud(points, colours=colors)


@as_numpy
def o3d_cloud(points, colour=None, colours=None, normals=None):

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    if colour is not None:
        return cloud.paint_uniform_color(colour)
    elif colours is not None:
        cloud.colors = o3d.utility.Vector3dVector(colours)
        return cloud

    return cloud.paint_uniform_color([1, 0, 0])


@as_numpy
def o3d_clouds(batch_points, colour=None, colours=None, p_colours=None):
    # KN#
    # colour -> paint all the same colour
    # colours -> colour for each point cloud
    # p_colours -> points for each point in each cloud
    if colour is not None:
        return [o3d_cloud(p, colour) for p in zip(batch_points)]

    if colours is not None:
        return [o3d_cloud(p, c) for p, c in zip(batch_points, colours)]

    if p_colours is not None:
        return [o3d_cloud(p, colours=c) for p, c in zip(batch_points, p_colours)]

    return [o3d_cloud(p, n) for p, n in zip(batch_points)]


@as_numpy
def o3d_voxel_grid(
    width: float,
    depth: float,
    height: float,
    voxel_size: float,
    origin=np.asarray([0, 0, 0]),
    colour=np.asarray([1, 1, 0]),
):
    return o3d.geometry.VoxelGrid.create_dense(
        origin, colour, voxel_size, width, depth, height
    )


@as_numpy
def o3d_cylinder(radius, length, colour=(1, 0, 0)):
    return o3d.geometry.TriangleMesh.create_cylinder(
        radius, length
    ).paint_uniform_color(colour)


@as_numpy
def o3d_cylinders(radii, length, colour=None, colours=None):
    cylinders = [o3d_cylinder(r, l, colour) for r, l in zip(radii, length)]
    return paint_o3d_geoms(cylinders, colour, colours)


def paint_o3d_geoms(geometries, colour=None, colours=None):
    if colours is not None:
        return [
            geom.paint_uniform_color(colours[i]) for i, geom in enumerate(geometries)
        ]
    elif colour is not None:
        return [geom.paint_uniform_color(colour) for geom in geometries]
    return geometries


@as_numpy
def unit_circle(n):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return np.stack([np.sin(a), np.cos(a)], axis=1)


@as_numpy
def cylinder_triangles(m, n):
    tri1 = np.array([0, 1, 2])
    tri2 = np.array([2, 3, 0])

    v0 = np.arange(m)
    v1 = (v0 + 1) % m
    v2 = v1 + m
    v3 = v0 + m

    edges = np.stack([v0, v1, v2, v3], axis=1)

    segments = np.arange(n - 1) * m
    edges = edges.reshape(1, *edges.shape) + segments.reshape(n - 1, 1, 1)

    edges = edges.reshape(-1, 4)
    return np.concatenate([edges[:, tri1], edges[:, tri2]])


@as_numpy
def tube_vertices(points, radii, n=10):
    circle = unit_circle(n).astype(np.float32)

    dirs = vertex_dirs(points)
    t = gen_tangents(dirs, random_unit())

    b = np.stack([t, np.cross(t, dirs)], axis=1)
    b = b * radii.reshape(-1, 1, 1)

    return np.einsum("bdx,md->bmx", b, circle) + points.reshape(points.shape[0], 1, 3)


def o3d_lines_between_clouds(cld1, cld2):
    pts1 = np.asarray(cld1.points)
    pts2 = np.asarray(cld2.points)

    interweaved = np.hstack((pts1, pts2)).reshape(-1, 3)
    return o3d_line_set(
        interweaved, np.arange(0, min(pts1.shape[0], pts2.shape[0]) * 2).reshape(-1, 2)
    )


@as_numpy
def o3d_tube_mesh(points, radii, colour=(1, 0, 0), n=10):
    points = points.cpu().numpy()
    radii = radii.cpu().numpy()
    points = tube_vertices(points, radii, n)

    n, m, _ = points.shape
    indexes = cylinder_triangles(m, n)

    mesh = o3d_mesh(points.reshape(-1, 3), indexes)
    mesh.compute_vertex_normals()

    return mesh.paint_uniform_color(colour)


def sample_o3d_lineset(lineset, sample_rate):
    edges = np.asarray(lineset.lines)
    xyz = np.asarray(lineset.points)

    pts, radius = [], []

    for i, edge in enumerate(edges):
        start = xyz[edge[0]]
        end = xyz[edge[1]]

        v = end - start
        length = np.linalg.norm(v)
        direction = v / length
        num_points = np.ceil(length / sample_rate)

        if int(num_points) > 0.0:
            spaced_points = np.arange(
                0, float(length), step=float(length / num_points)
            ).reshape(-1, 1)
            pts.append(start + direction * spaced_points)

    return np.concatenate(pts, axis=0)


def o3d_elipsoid(semi_axes=(1.0, 1.0, 1.0), position=(0.0, 0.0, 0.0), num_segments=32):
    """
    Create vertices and triangles for an ellipsoid mesh.

    Parameters:
        semi_axes (tuple): Lengths of the semi-axes (a, b, c) of the ellipsoid.
        num_segments (int): Number of segments for both horizontal and vertical divisions.

    Returns:
        vertices (numpy.ndarray): Array of vertex coordinates.
        triangles (numpy.ndarray): Array of triangle vertex indices.
    """
    a, b, c = semi_axes

    u = np.linspace(0, 2 * np.pi, num_segments)
    v = np.linspace(0, np.pi, num_segments)

    # Create a mesh grid of points
    u, v = np.meshgrid(u, v)
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    # Convert the mesh grid into vertices
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    # Create triangles from the vertices
    i, j = np.meshgrid(range(num_segments - 1), range(num_segments - 1))
    i, j = i.flatten(), j.flatten()
    a = i * num_segments + j
    b = a + 1
    c = (i + 1) * num_segments + j
    d = c + 1

    # Create two triangles from each quad
    triangles = np.column_stack((a, b, c, b, d, c)).reshape(-1, 3)

    return o3d_mesh(vertices + position, triangles)


def o3d_semi_ellipsoid(semi_axes=(1.0, 1.0, 1.0), num_segments=32, num_slices=16):
    """
    Create vertices and triangles for a semi-ellipsoid mesh without using loops.

    Parameters:
        semi_axes (tuple): Lengths of the semi-axes (a, b, c) of the semi-ellipsoid.
        num_segments (int): Number of segments around the circumference.
        num_slices (int): Number of slices along the height.

    Returns:
        vertices (numpy.ndarray): Array of vertex coordinates.
        triangles (numpy.ndarray): Array of triangle vertex indices.
    """
    a, b, c = semi_axes

    u = np.linspace(0, np.pi, num_segments)
    v = np.linspace(0, np.pi / 2, num_slices)

    u, v = np.meshgrid(u, v)

    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    # Convert the mesh grid into vertices
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    i, j = np.meshgrid(np.arange(num_slices - 1), np.arange(num_segments - 1))
    i, j = i.flatten(), j.flatten()

    a = i * num_segments + j
    b = a + 1
    c = (i + 1) * num_segments + j
    d = c + 1

    triangles = np.column_stack((a, b, c, b, d, c)).reshape(-1, 3)

    return o3d_mesh(vertices, triangles.astype(np.uint32))
