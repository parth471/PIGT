"""
check_data.py — quick exploratory look at the raw node table.
Run this right after build_nodes.py, before building the graph.
"""

import pandas as pd

nodes = pd.read_csv("data/processed_nodes.csv", parse_dates=["time"])

print("Shape:", nodes.shape)
print("\nTime range:", nodes["time"].min(), "to", nodes["time"].max())
print("Number of unique nodes:", nodes["node_id"].nunique())
print("\nMissing values per column:\n", nodes.isna().sum())
print("\nSummary stats:\n", nodes.describe())
