import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# -----------------------------------------
# Load data
# -----------------------------------------

features = np.load("data/node_features.npy")
edges = pd.read_csv("data/edges.csv")


# -----------------------------------------
# Select first day
# -----------------------------------------

# Shape: (nodes, features)
x = features[0]

print("Node feature shape:", x.shape)


# -----------------------------------------
# Convert node features to PyTorch tensor
# -----------------------------------------

x = torch.tensor(
    x,
    dtype=torch.float32
)


# -----------------------------------------
# Build edge_index
# -----------------------------------------

edge_index = torch.tensor(
    edges[["source", "target"]].values.T,
    dtype=torch.long
)


# -----------------------------------------
# Edge distance
# -----------------------------------------

edge_attr = torch.tensor(
    edges["distance_km"].values,
    dtype=torch.float32
).view(-1, 1)


# -----------------------------------------
# Create PyG graph
# -----------------------------------------

graph = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)


# -----------------------------------------
# Print information
# -----------------------------------------

print("\nGraph:")
print(graph)

print("\nNumber of nodes:", graph.num_nodes)
print("Number of edges:", graph.num_edges)
print("Node features:", graph.num_node_features)
print("Edge features:", graph.num_edge_features)