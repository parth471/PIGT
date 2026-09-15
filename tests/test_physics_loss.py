from pathlib import Path
import sys
import torch

# Ensure src/ is on python path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physics.physics_loss import OceanAdvectionDiffusionLoss


def test_ocean_physics_loss():
    print("=" * 70)
    print("TEST: 2D Ocean Thermal Advection-Diffusion Loss")
    print("=" * 70)

    B = 2
    T = 7
    N = 12204
    F = 13
    E = 97632

    # Synthetic input history: [B, T, N, F]
    x_features = torch.randn(B, T, N, F, dtype=torch.float32, requires_grad=True)

    # Next-day SST prediction from model: [B, N]
    pred_sst_next = torch.randn(B, N, dtype=torch.float32, requires_grad=True)

    # Graph edge indices: [2, E]
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long)

    # Synthetic coordinates (lat: 10-20, lon: 80-90)
    lat = torch.linspace(10, 20, N)
    lon = torch.linspace(80, 90, N)
    coords = torch.stack([lat, lon], dim=-1)

    loss_fn = OceanAdvectionDiffusionLoss(diffusivity_kappa=1e-3, delta_t_days=1.0)

    # 1. Forward Pass
    loss = loss_fn(
        pred_sst_next=pred_sst_next,
        x_features=x_features,
        edge_index=edge_index,
        coords=coords,
    )

    print(f"Computed Physics Residual Loss: {loss.item():.6f}")

    # Shape and numerical assertions
    assert loss.dim() == 0, "Loss must be scalar"
    assert torch.isfinite(loss), "Physics loss must be finite"
    assert loss.item() >= 0.0, "Physics loss (squared residual) must be non-negative"

    # 2. Backward Pass (Gradient Flow)
    loss.backward()

    assert pred_sst_next.grad is not None, "Gradients must propagate to predicted SST"
    assert torch.isfinite(pred_sst_next.grad).all(), "Gradients must be finite"

    print("Gradient backpropagation: OK")
    print("\n✓ test_ocean_physics_loss: PASSED")


if __name__ == "__main__":
    test_ocean_physics_loss()
