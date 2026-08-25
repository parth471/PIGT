"""
STEP 5 of 8 — create_pyg_graph.py
====================================================================
Combines data/node_features.csv + data/edges.csv into a single
torch_geometric.data.Data object -- the standard format every PyG
GNN layer (GCNConv, GATConv, etc.) expects.

A PyG Data object has three key pieces:
    x           : (num_nodes, num_features) node feature matrix
    edge_index  : (2, num_edges) which nodes are connected
    edge_weight : (num_edges,) optional edge strength (we use inverse distance)

Output: data/graph.pt
"""

import pandas as pd
import torch
from torch_geometric.data import Data

NODES_PATH = "data/node_features.csv"
EDGES_PATH = "data/edges.csv"
OUTPUT_PATH = "data/graph.pt"
FEATURE_COLS = ["thetao", "so", "zos", "uo", "vo"]
TARGET_COL = "thetao"   # the variable this GNN will learn to predict at masked nodes


def main():
    nodes = pd.read_csv(NODES_PATH).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(EDGES_PATH)

    # sanity: node_id must be a clean 0..N-1 range so it lines up with row order
    assert (nodes["node_id"].values == range(len(nodes))).all(), \
        "node_id is not a contiguous 0..N-1 range — reindex before building the graph"

    x = torch.tensor(nodes[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(nodes[TARGET_COL].values, dtype=torch.float32)

    edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)
    edge_weight = torch.tensor(edges["weight"].values, dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    graph.pos = torch.tensor(nodes[["lat", "lon"]].values, dtype=torch.float32)  # keep for plotting later

    torch.save(graph, OUTPUT_PATH)
    print(f"[INFO] Graph object: {graph}")
    print(f"[INFO] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
