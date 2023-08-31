import cudf
import cugraph
import numpy as np
import open3d as o3d
import pytest
import torch

from smart_tree.data_types.graph import Graph


# Define a fixture for a sample Graph instance
@pytest.fixture
def sample_graph():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
    edge_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return Graph(vertices, edges, edge_weights)


# Test the 'to_o3d_lineset' method
def test_to_o3d_lineset(sample_graph):
    lineset = sample_graph.to_o3d_lineset(colour=(1, 0, 0))

    # Ensure the returned object is of the correct type
    assert isinstance(lineset, o3d.geometry.LineSet)


# Test the 'to_device' method
def test_to_device(sample_graph):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_graph = sample_graph.to_device(device)

    # Check if the vertices, edges, and edge_weights are on the target device and have the correct dtype
    assert device_graph.vertices.device == device
    assert device_graph.vertices.dtype == torch.float32
    assert device_graph.edges.device == device
    assert device_graph.edges.dtype == torch.int64
    assert device_graph.edge_weights.device == device
    assert device_graph.edge_weights.dtype == torch.float32


# Test the 'connected_cugraph_components' method
def test_connected_cugraph_components(sample_graph):
    # Ensure that the method returns a list of cugraph.Graph objects
    components = sample_graph.connected_cugraph_components(minimum_vertices=1)
    assert isinstance(components, list)
    assert all(isinstance(graph, cugraph.Graph) for graph in components)
