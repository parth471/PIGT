import torch

from pigt_model import PIGTModel


def main():

    B = 2
    T = 7
    N = 12204
    F = 13

    print("=" * 70)
    print("PIGT — COMPLETE MODEL TEST")
    print("=" * 70)

    # ----------------------------------------------------
    # Dummy input
    # ----------------------------------------------------

    x = torch.randn(
        B,
        T,
        N,
        F,
        dtype=torch.float32,
    )

    # ----------------------------------------------------
    # Dummy graph
    # ----------------------------------------------------

    num_edges = 97632

    edge_index = torch.randint(
        0,
        N,
        (2, num_edges),
        dtype=torch.long,
    )

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    model = PIGTModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
    )

    print("\nInput:")
    print(tuple(x.shape))

    print("\nGraph:")
    print(tuple(edge_index.shape))

    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------

    prediction = model(
        x,
        edge_index,
    )

    print("\nPrediction:")
    print(tuple(prediction.shape))

    # ----------------------------------------------------
    # Expected output
    # ----------------------------------------------------

    assert prediction.shape == (
        B,
        N,
    )

    assert torch.isfinite(
        prediction
    ).all()

    # ----------------------------------------------------
    # Dummy target
    # ----------------------------------------------------

    target = torch.randn(
        B,
        N,
    )

    loss = torch.mean(
        (prediction - target) ** 2
    )

    print(
        "\nLoss:",
        loss.item(),
    )

    # ----------------------------------------------------
    # Backward
    # ----------------------------------------------------

    loss.backward()

    print(
        "Backward pass: OK"
    )

    print(
        "\nCOMPLETE PIGT MODEL TEST PASSED"
    )


if __name__ == "__main__":
    main()