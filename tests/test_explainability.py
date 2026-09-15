from pathlib import Path
import sys
import numpy as np
import torch

# Ensure src/ is on python path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pigt_uncertainty import PIGTUncertaintyModel
from explainability.temporal_attention import TemporalAttentionExtractor
from explainability.visualize_results import (
    plot_temporal_attention_profile,
    plot_spatial_field,
    plot_forecast_uncertainty_bounds,
)


def test_explainability():
    print("=" * 70)
    print("TEST: Explainability & Attention Visualization")
    print("=" * 70)

    B = 1
    T = 7
    N = 500  # Subsample for fast test
    F = 13
    E = 4000

    x = torch.randn(B, T, N, F, dtype=torch.float32)
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long)

    model = PIGTUncertaintyModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
    )

    extractor = TemporalAttentionExtractor(model)
    attn_map = extractor.compute_manual_attention_map(x, edge_index)

    print(f"Attention Map shape: {tuple(attn_map.shape)}")
    assert attn_map.shape == (B, N, T, T)

    # Check softmax property (rows sum to 1)
    row_sums = attn_map.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    lag_profile = extractor.get_lag_attribution(attn_map)
    print(f"7-Day Normalized Lag Profile: {lag_profile}")
    assert len(lag_profile) == 7
    assert np.isclose(lag_profile.sum(), 1.0)

    # Test visualization functions output
    test_output_dir = Path(__file__).resolve().parents[1] / "results" / "test_plots"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    plot_temporal_attention_profile(
        lag_profile,
        save_path=test_output_dir / "test_attention_profile.png",
    )

    lats = np.linspace(10, 20, N)
    lons = np.linspace(80, 90, N)
    test_values = np.random.rand(N)

    plot_spatial_field(
        lats,
        lons,
        test_values,
        title="Test Spatial Heatmap",
        colorbar_label="Value",
        save_path=test_output_dir / "test_spatial_map.png",
    )

    plot_forecast_uncertainty_bounds(
        y_true=np.random.randn(N),
        y_pred=np.random.randn(N),
        std=np.abs(np.random.randn(N)),
        save_path=test_output_dir / "test_uq_bounds.png",
    )

    assert (test_output_dir / "test_attention_profile.png").exists()
    assert (test_output_dir / "test_spatial_map.png").exists()
    assert (test_output_dir / "test_uq_bounds.png").exists()

    print("\n✓ test_explainability: PASSED")


if __name__ == "__main__":
    test_explainability()
