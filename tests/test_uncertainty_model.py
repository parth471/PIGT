from pathlib import Path
import sys
import torch

# Ensure src/ is on python path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pigt_uncertainty import PIGTUncertaintyModel, GaussianNLLLoss


def test_pigt_uncertainty():
    print("=" * 70)
    print("TEST: PIGT Model with Uncertainty Quantification & MC Dropout")
    print("=" * 70)

    B = 2
    T = 7
    N = 12204
    F = 13
    E = 97632

    x = torch.randn(B, T, N, F, dtype=torch.float32)
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long)
    target = torch.randn(B, N, dtype=torch.float32)

    model = PIGTUncertaintyModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.1,
    )

    # 1. Forward Pass (Mean and Log-Variance)
    mean, log_var = model(x, edge_index)

    print(f"Output Mean shape:    {tuple(mean.shape)}")
    print(f"Output Log-Var shape: {tuple(log_var.shape)}")

    assert mean.shape == (B, N)
    assert log_var.shape == (B, N)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()

    # 2. Gaussian NLL Loss & Backward
    nll_loss_fn = GaussianNLLLoss()
    loss = nll_loss_fn(mean, log_var, target)

    print(f"Gaussian NLL Loss:   {loss.item():.6f}")

    assert torch.isfinite(loss)
    loss.backward()

    # 3. MC Dropout Uncertainty Inference
    uq_results = model.predict_with_uncertainty(x, edge_index, num_mc_samples=5)

    print("\nMC Dropout UQ Decomposition:")
    print(f"  - Predictive Mean shape: {tuple(uq_results['mean'].shape)}")
    print(f"  - Aleatoric Var shape:   {tuple(uq_results['aleatoric_var'].shape)}")
    print(f"  - Epistemic Var shape:   {tuple(uq_results['epistemic_var'].shape)}")
    print(f"  - Total Std shape:       {tuple(uq_results['std'].shape)}")

    assert (uq_results["aleatoric_var"] >= 0).all()
    assert (uq_results["epistemic_var"] >= 0).all()
    assert (uq_results["std"] >= 0).all()

    print("\n✓ test_pigt_uncertainty: PASSED")


if __name__ == "__main__":
    test_pigt_uncertainty()
