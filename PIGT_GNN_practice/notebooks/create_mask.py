"""
STEP 6 of 8 — create_mask.py
====================================================================
This is the piece that defines the actual task. Instead of splitting by
TIME (like forecasting does), we split by NODE: some grid points are
"known" (train), some are held back to check generalization (val/test),
all at the SAME point in time.

Why this framing: it simulates a real gap-filling problem -- e.g. "I have
buoys/satellite readings at 70% of locations, can the GNN infer the rest
from spatial context alone?"

Output: data/masks.pt  (boolean tensors: train_mask, val_mask, test_mask)
"""

import torch

GRAPH_PATH = "data/graph.pt"
OUTPUT_PATH = "data/masks.pt"
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
SEED = 42


def main():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    num_nodes = graph.num_nodes

    torch.manual_seed(SEED)
    perm = torch.randperm(num_nodes)

    train_end = int(num_nodes * TRAIN_FRAC)
    val_end = int(num_nodes * (TRAIN_FRAC + VAL_FRAC))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    torch.save({"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}, OUTPUT_PATH)
    print(f"[INFO] {num_nodes} nodes total -> "
          f"train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}")
    print(f"[INFO] Saved masks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
