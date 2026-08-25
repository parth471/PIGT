"""
STEP 2 of 8 — build_edges.py
====================================================================
Reads data/processed_nodes.csv, takes the unique (lat, lon) locations
(one graph, shared across all timesteps), and connects each node to its
k nearest neighbors by great-circle distance.

Output: data/edges.csv
    columns: src, dst, weight   (weight = inverse distance, closer = stronger)
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

INPUT_PATH = "data/processed_nodes.csv"
OUTPUT_PATH = "data/edges.csv"
K_NEIGHBORS = 6


def main():
    nodes = pd.read_csv(INPUT_PATH)
    unique_nodes = nodes.drop_duplicates(subset="node_id")[["node_id", "lat", "lon"]].sort_values("node_id")

    lat_rad = np.radians(unique_nodes["lat"].values)
    lon_rad = np.radians(unique_nodes["lon"].values)
    R = 6371.0
    x = R * lon_rad * np.cos(lat_rad.mean())
    y = R * lat_rad
    xy = np.stack([x, y], axis=1)

    tree = cKDTree(xy)
    dist, neighbor_idx = tree.query(xy, k=K_NEIGHBORS + 1)  # +1 includes self

    src, dst, weight = [], [], []
    node_ids = unique_nodes["node_id"].values
    for i in range(len(node_ids)):
        for rank in range(1, K_NEIGHBORS + 1):  # skip self at rank 0
            j = neighbor_idx[i, rank]
            d = dist[i, rank]
            src.append(node_ids[i])
            dst.append(node_ids[j])
            weight.append(1.0 / (d + 1e-6))

    edges = pd.DataFrame({"src": src, "dst": dst, "weight": weight})
    edges.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Built graph: {len(unique_nodes)} nodes, {len(edges)} directed edges "
          f"(k={K_NEIGHBORS}). Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
