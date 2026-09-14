import torch

from temporal_transformer import TemporalTransformer


def main():

    # Dummy representation after GNN
    #
    # B = 2
    # T = 7
    # N = 12204
    # H = 64

    x = torch.randn(
        2,
        7,
        12204,
        64,
    )

    model = TemporalTransformer(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
    )

    output = model(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    assert output.shape == (
        2,
        12204,
        64,
    )

    assert torch.isfinite(output).all()

    print("Temporal Transformer test: PASSED")


if __name__ == "__main__":
    main()