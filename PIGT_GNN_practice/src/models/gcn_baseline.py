"""
STEP 8 of 8 — gcn_baseline.py
====================================================================
The actual GNN. Uses torch_geometric's built-in GCNConv (production-
standard layer, same math as the by-hand version from earlier practice:
H' = sigma( D^-1/2 * A_hat * D^-1/2 * H * W ), but implemented and
optimized by the library).

Task: node regression. Predict `thetao` (SST) at held-out nodes, using
only their position in the graph + the OTHER features (so, zos, uo, vo)
-- NOT thetao itself, since that would be cheating (the model would just
copy the answer).

Run order: build_nodes -> build_edges -> build_features -> check_features
           -> create_pyg_graph -> create_mask -> (this file)
"""

import sys
sys.path.insert(0, "src/training")
from prepare_data import load_training_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64            # was 32 -- real data has more structure to capture than the synthetic test
DROPOUT = 0.2              # was 0.3 -- slightly less, since the real problem was underfitting not overfitting
LR = 5e-3                  # was 1e-2 -- more stable on a much bigger graph (14k vs 400 nodes)
EPOCHS = 300                # was 100 -- more optimization steps needed at this scale
TARGET_FEATURE_INDEX = 0   # index of `thetao` within FEATURE_COLS in create_pyg_graph.py


class GCNBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        return self.out(h).squeeze(-1)


def main():
    graph = load_training_graph()

    # mask out the target variable from the INPUT features so the model can't just
    # copy it -- this is the single most common mistake in a first GNN attempt
    x = graph.x.clone()
    target_col = x[:, TARGET_FEATURE_INDEX].clone()
    x[:, TARGET_FEATURE_INDEX] = 0.0

    x = x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    # normalize edge weights to [0, 1] -- raw inverse-distance weights can sit in
    # an awkward range depending on the region's coordinate scale, which makes
    # GCNConv's internal degree-normalization behave inconsistently
    edge_weight_raw = graph.edge_attr.to(DEVICE)
    edge_weight = edge_weight_raw / edge_weight_raw.max()
    y = target_col.to(DEVICE)
    train_mask = graph.train_mask.to(DEVICE)
    val_mask = graph.val_mask.to(DEVICE)
    test_mask = graph.test_mask.to(DEVICE)

    model = GCNBaseline(in_dim=x.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(x, edge_index, edge_weight)
        loss = F.mse_loss(pred[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index, edge_weight)
                val_mse = F.mse_loss(pred[val_mask], y[val_mask]).item()
                # correlation is the real tell: MSE can look okay even when the
                # model has just collapsed to predicting the mean everywhere
                val_corr = torch.corrcoef(torch.stack([pred[val_mask], y[val_mask]]))[0, 1].item()
            print(f"[Epoch {epoch+1:03d}/{EPOCHS}] train_mse={loss.item():.4f}  "
                  f"val_mse={val_mse:.4f}  val_corr={val_corr:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index, edge_weight)
        test_mse = F.mse_loss(pred[test_mask], y[test_mask]).item()
        test_mae = F.l1_loss(pred[test_mask], y[test_mask]).item()
        test_corr = torch.corrcoef(torch.stack([pred[test_mask], y[test_mask]]))[0, 1].item()
    print(f"[RESULTS] test_mse={test_mse:.4f}  test_mae={test_mae:.4f}  test_corr={test_corr:.4f}")

    torch.save(model.state_dict(), "data/gcn_baseline_model.pt")
    print("[INFO] Saved model weights to data/gcn_baseline_model.pt")
    return model


if __name__ == "__main__":
    main()