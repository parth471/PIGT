"""
STEP 4 of 8 — check_features.py
====================================================================
Always run this before training. Catching a bad graph or NaN-filled
feature column here saves an hour of debugging a model that "just won't
learn" for no apparent reason.
"""

import pandas as pd

NODES_PATH = "data/node_features.csv"
EDGES_PATH = "data/edges.csv"


def main():
    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    print("=== NODE FEATURES ===")
    print(f"Shape: {nodes.shape}")
    print(f"NaN counts:\n{nodes.isna().sum()}")
    print(f"Value ranges:\n{nodes.describe().loc[['min','max','mean','std']]}")

    print("\n=== EDGES ===")
    print(f"Shape: {edges.shape}")
    print(f"Unique src nodes: {edges['src'].nunique()} / total nodes: {len(nodes)}")
    isolated = set(nodes["node_id"]) - set(edges["src"]).union(set(edges["dst"]))
    print(f"Isolated nodes (no edges at all): {len(isolated)}")
    if isolated:
        print(f"  -> {list(isolated)[:10]}{'...' if len(isolated) > 10 else ''}")

    print("\n=== VERDICT ===")
    problems = []
    if nodes.isna().sum().sum() > 0:
        problems.append("NaNs present in features — re-run build_features.py / check interpolation")
    if len(isolated) > 0:
        problems.append("isolated nodes found — increase K_NEIGHBORS in build_edges.py")
    if not problems:
        print("Looks clean. Safe to proceed to create_pyg_graph.py")
    else:
        for p in problems:
            print(f"  [WARNING] {p}")


if __name__ == "__main__":
    main()
