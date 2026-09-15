from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"


# ============================================================
# Model
# ============================================================

class GCNBaseline(nn.Module):
    """
    Spatial GCN baseline.

    Input:
        [N, F]

    Output:
        [N, 1]

    The model predicts next-day SST for every ocean node.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gcn1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.gcn2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        self.dropout = dropout

        self.output_layer = nn.Linear(
            hidden_channels,
            1,
        )

    def forward(
        self,
        x,
        edge_index,
    ):
        x = self.gcn1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.gcn2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.output_layer(x)

        return x.squeeze(-1)


# ============================================================
# Load one snapshot
# ============================================================

def load_graph():
    if GRAPH_PATH.exists():
        graph = torch.load(
            GRAPH_PATH,
            map_location="cpu",
        )
        edge_index = graph["edge_index"]
    else:
        print("Graph file not found at", GRAPH_PATH, "— using synthetic edge_index (2, 97632)")
        edge_index = torch.randint(0, 12204, (2, 97632), dtype=torch.long)

    return edge_index


def load_data():
    x_train_path = DATA_DIR / "X_train.npy"
    y_train_path = DATA_DIR / "Y_train.npy"

    if x_train_path.exists() and y_train_path.exists():
        X_train = np.load(x_train_path)
        Y_train = np.load(y_train_path)
        X_last = X_train[:, -1, :, :]
        x = torch.tensor(X_last[0], dtype=torch.float32)
        y = torch.tensor(Y_train[0, :, 0], dtype=torch.float32)
    else:
        print("Data files not found in", DATA_DIR, "— using synthetic input (12204, 13) and target (12204)")
        x = torch.randn(12204, 13, dtype=torch.float32)
        y = torch.randn(12204, dtype=torch.float32)

    return x, y


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 70)
    print("PIGT — GCN BASELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    x, y = load_data()

    print(
        "Input tensor:",
        tuple(x.shape),
    )

    print(
        "Target tensor:",
        tuple(y.shape),
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    edge_index = load_graph()

    print(
        "Edge index:",
        tuple(edge_index.shape),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GCNBaseline(
        in_channels=13,
        hidden_channels=64,
    )

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    prediction = model(
        x,
        edge_index,
    )

    print(
        "\nPrediction shape:",
        tuple(prediction.shape),
    )

    print(
        "Target shape:",
        tuple(y.shape),
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = F.mse_loss(
        prediction,
        y,
    )

    print(
        "\nInitial MSE:",
        loss.item(),
    )

    # --------------------------------------------------------
    # Backward test
    # --------------------------------------------------------

    loss.backward()

    print(
        "Backward pass: OK"
    )

    print(
        "\nGCN BASELINE TEST PASSED"
    )


if __name__ == "__main__":
    main()