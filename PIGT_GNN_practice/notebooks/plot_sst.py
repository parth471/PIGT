"""
plot_sst.py — visualize predicted vs. true SST spatially at the test nodes.
Run this after gcn_baseline.py has produced data/gcn_baseline_model.pt.
"""

import sys
sys.path.insert(0, "src/training")
sys.path.insert(0, "src/models")
from prepare_data import load_training_graph
from gcn_baseline import GCNBaseline, TARGET_FEATURE_INDEX

import torch
import matplotlib.pyplot as plt

graph = load_training_graph()
x = graph.x.clone()
true_sst = x[:, TARGET_FEATURE_INDEX].clone()
x[:, TARGET_FEATURE_INDEX] = 0.0

model = GCNBaseline(in_dim=x.shape[1])
model.load_state_dict(torch.load("data/gcn_baseline_model.pt"))
model.eval()

with torch.no_grad():
    pred_sst = model(x, graph.edge_index, graph.edge_attr)

lat = graph.pos[:, 0].numpy()
lon = graph.pos[:, 1].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sc0 = axes[0].scatter(lon, lat, c=true_sst.numpy(), cmap="coolwarm")
axes[0].set_title("True SST (normalized)")
plt.colorbar(sc0, ax=axes[0])

sc1 = axes[1].scatter(lon, lat, c=pred_sst.numpy(), cmap="coolwarm")
axes[1].set_title("Predicted SST (normalized)")
plt.colorbar(sc1, ax=axes[1])

test_mask = graph.test_mask.numpy()
sc2 = axes[2].scatter(lon[test_mask], lat[test_mask],
                       c=(pred_sst.numpy() - true_sst.numpy())[test_mask], cmap="RdBu_r")
axes[2].set_title("Error on TEST nodes only")
plt.colorbar(sc2, ax=axes[2])

plt.tight_layout()
plt.savefig("data/sst_prediction_plot.png", dpi=120)
print("[INFO] Saved plot to data/sst_prediction_plot.png")
