from pathlib import Path

import numpy as np
import torch

from pigt_model import PIGTModel


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"


def main():

    print("=" * 70)
    print("PIGT — REAL GRAPH TEST")
    print("=" * 70)

    # ----------------------------------------------------
    # Load one real training sample
    # ----------------------------------------------------

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    Y_train = np.load(
        DATA_DIR / "Y_train.npy"
    )

    # Use only the first training sample
    x = torch.tensor(
        X_train[0:1],
        dtype=torch.float32,
    )

    # SST is target channel 0
    y = torch.tensor(
        Y_train[0:1, :, 0],
        dtype=torch.float32,
    )

    print("\nInput:")
    print(tuple(x.shape))

    print("Target:")
    print(tuple(y.shape))

    # ----------------------------------------------------
    # Load actual graph
    # ----------------------------------------------------

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
    )

    edge_index = graph["edge_index"]

    print("\nGraph:")
    print("Nodes :", graph["num_nodes"])
    print("Edges :", edge_index.shape[1])
    print("Edge index:", tuple(edge_index.shape))

    # ----------------------------------------------------
    # Create model
    # ----------------------------------------------------

    model = PIGTModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
    )

    # ----------------------------------------------------
    # Forward pass
    # ----------------------------------------------------

    prediction = model(
        x,
        edge_index,
    )

    print("\nPrediction:")
    print(tuple(prediction.shape))

    # ----------------------------------------------------
    # Shape and numerical checks
    # ----------------------------------------------------

    assert prediction.shape == (
        1,
        12204,
    )

    assert torch.isfinite(
        prediction
    ).all()

    # ----------------------------------------------------
    # Calculate loss
    # ----------------------------------------------------

    loss = torch.mean(
        (prediction - y) ** 2
    )

    print(
        "\nInitial real-data MSE:",
        loss.item(),
    )

    # ----------------------------------------------------
    # Backward pass
    # ----------------------------------------------------

    loss.backward()

    print(
        "Backward pass: OK"
    )

    print(
        "\nREAL GRAPH TEST PASSED"
    )


if __name__ == "__main__":
    main()