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

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
    )

    edge_index = graph["edge_index"]

    return edge_index


def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    Y_train = np.load(
        DATA_DIR / "Y_train.npy"
    )

    return X_train, Y_train


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

    X_train, Y_train = load_data()

    print("\nDataset:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)

    # --------------------------------------------------------
    # We use the most recent historical timestep
    #
    # X:
    # [samples, 7, nodes, 13]
    #
    # Select:
    # [samples, nodes, 13]
    # --------------------------------------------------------

    X_last = X_train[:, -1, :, :]

    print(
        "\nLatest timestep:",
        X_last.shape,
    )

    # --------------------------------------------------------
    # One sample for model test
    # --------------------------------------------------------

    x = torch.tensor(
        X_last[0],
        dtype=torch.float32,
    )

    # Target SST only
    y = torch.tensor(
        Y_train[0, :, 0],
        dtype=torch.float32,
    )

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
        float(loss),
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