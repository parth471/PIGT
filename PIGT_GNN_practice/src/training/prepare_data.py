"""
STEP 7 of 8 — prepare_data.py
====================================================================
Loads the graph (create_pyg_graph.py output) and the masks
(create_mask.py output), attaches the masks onto the graph object, and
returns one ready-to-train PyG Data object. This is the function
gcn_baseline.py imports.
"""

import torch


def load_training_graph(graph_path="data/graph.pt", masks_path="data/masks.pt"):
    graph = torch.load(graph_path, weights_only=False)
    masks = torch.load(masks_path, weights_only=False)

    graph.train_mask = masks["train_mask"]
    graph.val_mask = masks["val_mask"]
    graph.test_mask = masks["test_mask"]
    return graph


if __name__ == "__main__":
    g = load_training_graph()
    print(g)
    print(f"train/val/test node counts: "
          f"{g.train_mask.sum().item()}/{g.val_mask.sum().item()}/{g.test_mask.sum().item()}")
