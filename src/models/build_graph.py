from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final"
GRAPH_DIR = ROOT / "data" / "graph"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

K_NEIGHBORS = 8


def main():
    print("=" * 70)
    print("PIGT — BUILD FINAL 12,204-NODE GRAPH")
    print("=" * 70)

    # --------------------------------------------------------
    # Load final node coordinates
    # --------------------------------------------------------

    lat = np.load(DATA_DIR / "node_lat.npy")
    lon = np.load(DATA_DIR / "node_lon.npy")

    print("\nLoaded node coordinates:")
    print("Latitude shape :", lat.shape)
    print("Longitude shape:", lon.shape)

    # --------------------------------------------------------
    # Basic consistency checks
    # --------------------------------------------------------

    if lat.shape != lon.shape:
        raise ValueError(
            f"Latitude/longitude shape mismatch: "
            f"{lat.shape} vs {lon.shape}"
        )

    if lat.ndim != 1:
        raise ValueError(
            f"Expected 1D node coordinates, got {lat.ndim}D"
        )

    n_nodes = len(lat)

    print("Number of nodes:", n_nodes)

    if n_nodes != 12204:
        raise ValueError(
            f"Expected 12,204 nodes, but found {n_nodes}"
        )

    # --------------------------------------------------------
    # Build coordinate matrix
    # --------------------------------------------------------

    coordinates = np.column_stack((lat, lon))

    # --------------------------------------------------------
    # k-nearest-neighbour graph
    # --------------------------------------------------------

    print(
        f"\nBuilding kNN graph with k={K_NEIGHBORS}..."
    )

    knn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1,
        algorithm="ball_tree",
    )

    knn.fit(coordinates)

    distances, indices = knn.kneighbors(coordinates)

    # First neighbour is the node itself.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # --------------------------------------------------------
    # Convert to edge_index format
    # --------------------------------------------------------

    source_nodes = np.repeat(
        np.arange(n_nodes),
        K_NEIGHBORS,
    )

    target_nodes = indices.reshape(-1)

    edge_index = np.stack(
        [source_nodes, target_nodes],
        axis=0,
    )

    # --------------------------------------------------------
    # Edge distances
    # --------------------------------------------------------

    edge_distance = distances.reshape(-1, 1)

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

    edge_index_tensor = torch.tensor(
        edge_index,
        dtype=torch.long,
    )

    edge_attr_tensor = torch.tensor(
        edge_distance,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    graph_path = GRAPH_DIR / "graph.pt"

    torch.save(
        {
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
            "num_nodes": n_nodes,
            "k": K_NEIGHBORS,
        },
        graph_path,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nGraph statistics:")
    print("Nodes:", n_nodes)
    print("Edges:", edge_index_tensor.shape[1])
    print(
        "Edge index shape:",
        tuple(edge_index_tensor.shape),
    )
    print(
        "Edge attribute shape:",
        tuple(edge_attr_tensor.shape),
    )

    print(
        "Mean coordinate distance:",
        float(edge_attr_tensor.mean()),
    )

    print(
        "Max coordinate distance:",
        float(edge_attr_tensor.max()),
    )

    print(
        "Min coordinate distance:",
        float(edge_attr_tensor.min()),
    )

    print(
        f"\nSaved graph to:\n{graph_path}"
    )

    print("\nGRAPH BUILD COMPLETE")


if __name__ == "__main__":
    main()