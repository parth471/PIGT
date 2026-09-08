"""
STEP 3 of 8 — build_features.py
====================================================================
Takes data/processed_nodes.csv and:
  1. Picks ONE snapshot in time (for the interpolation-style GNN task —
     "predict SST at masked nodes from their neighbors, at a single moment")
  2. Z-score normalizes each feature column
  3. Saves a clean node-feature matrix, ready for create_pyg_graph.py

If you'd rather do forecasting-style (state at t -> state at t+1) like the
earlier PIGT practice, see the commented alternative at the bottom — but for
matching your teammate's create_mask.py approach, single-snapshot is correct.

Output: data/node_features.csv
    columns: node_id, lat, lon, thetao, so, zos, uo, vo   (all normalized except lat/lon)
"""

import pandas as pd

INPUT_PATH = "data/processed_nodes.csv"
OUTPUT_PATH = "data/node_features.csv"
FEATURE_COLS = ["thetao", "so", "zos", "uo", "vo"]
SNAPSHOT_INDEX = -1   # which timestep to use as "today" (-1 = most recent)


def main():
    nodes = pd.read_csv(INPUT_PATH, parse_dates=["time"])
    unique_times = sorted(nodes["time"].unique())
    snapshot_time = unique_times[SNAPSHOT_INDEX]

    snapshot = nodes[nodes["time"] == snapshot_time].sort_values("node_id").reset_index(drop=True)
    print(f"[INFO] Using snapshot at time={snapshot_time} ({len(snapshot)} nodes).")

    for col in FEATURE_COLS:
        mu, sigma = snapshot[col].mean(), snapshot[col].std() + 1e-8
        snapshot[col] = (snapshot[col] - mu) / sigma
        print(f"[INFO] Normalized '{col}': mean={mu:.4f}, std={sigma:.4f}")

    snapshot[["node_id", "lat", "lon"] + FEATURE_COLS].to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved normalized node features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# ALTERNATIVE (forecasting-style, t -> t+1) — use instead of the above
# if your team decides to do next-step prediction rather than
# masked-node interpolation:
#
#   pivot = nodes.pivot(index="time", columns="node_id", values="thetao")
#   X = pivot.iloc[:-1].values   # state at t
#   Y = pivot.iloc[1:].values    # state at t+1
#
# This is what 04_gnn_preprocessing.py / 02_cmems_preprocessing.py did earlier.
