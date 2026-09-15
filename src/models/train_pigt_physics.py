from pathlib import Path
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure src/ is on path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pigt_uncertainty import PIGTUncertaintyModel, GaussianNLLLoss
from physics.physics_loss import OceanAdvectionDiffusionLoss
from explainability.temporal_attention import TemporalAttentionExtractor
from explainability.visualize_results import (
    plot_temporal_attention_profile,
    plot_spatial_field,
    plot_forecast_uncertainty_bounds,
)


# ============================================================
# Paths & Output Directories
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
FIGURES_DIR = ROOT / "results" / "figures"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Physics loss weight
LAMBDA_PHYSICS = 0.1
DIFFUSIVITY_KAPPA = 1e-3

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1
PATIENCE = 5
GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data Loaders (With Fallback for Testing)
# ============================================================

def load_data():
    x_train_p = DATA_DIR / "X_train.npy"
    y_train_p = DATA_DIR / "Y_train.npy"
    x_val_p = DATA_DIR / "X_val.npy"
    y_val_p = DATA_DIR / "Y_val.npy"

    if x_train_p.exists() and y_train_p.exists() and x_val_p.exists() and y_val_p.exists():
        print("✓ Loading real datasets from:", DATA_DIR)
        X_train = np.load(x_train_p, mmap_mode="r")
        Y_train = np.load(y_train_p, mmap_mode="r")
        X_val = np.load(x_val_p, mmap_mode="r")
        Y_val = np.load(y_val_p, mmap_mode="r")
    else:
        print("⚠ Dataset arrays not found — generating synthetic split for validation (Train: 10, Val: 4)")
        X_train = np.random.randn(10, 7, 12204, 13).astype(np.float32)
        Y_train = np.random.randn(10, 12204, 5).astype(np.float32)
        X_val = np.random.randn(4, 7, 12204, 13).astype(np.float32)
        Y_val = np.random.randn(4, 12204, 5).astype(np.float32)

    return X_train, Y_train, X_val, Y_val


def load_graph():
    if GRAPH_PATH.exists():
        print("✓ Loading graph from:", GRAPH_PATH)
        graph = torch.load(GRAPH_PATH, map_location="cpu")
        edge_index = graph["edge_index"].to(DEVICE)
    else:
        print("⚠ Graph file not found — creating synthetic edge index (2, 97632)")
        edge_index = torch.randint(0, 12204, (2, 97632), dtype=torch.long, device=DEVICE)

    return edge_index


# ============================================================
# Training Epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss_fn: nn.Module,
    phys_loss_fn: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    edge_index: torch.Tensor,
    lambda_phys: float = LAMBDA_PHYSICS,
) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_data_loss, total_phys_loss = 0.0, 0.0, 0.0
    num_samples = X_train.shape[0]

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start in range(0, num_samples, BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]

        x_b = torch.from_numpy(np.asarray(X_train[batch_idx])).float().to(DEVICE)
        y_b = torch.from_numpy(np.asarray(Y_train[batch_idx, :, 0])).float().to(DEVICE)  # SST channel 0

        optimizer.zero_grad(set_to_none=True)

        mean_pred, log_var_pred = model(x_b, edge_index)

        # 1. Data Loss (Gaussian NLL)
        loss_data = data_loss_fn(mean_pred, log_var_pred, y_b)

        # 2. Physics-Informed Advection-Diffusion Loss
        loss_phys = phys_loss_fn(mean_pred, x_b, edge_index)

        # 3. Composite Loss
        loss_total = loss_data + (lambda_phys * loss_phys)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        b_len = len(batch_idx)
        total_loss += loss_total.item() * b_len
        total_data_loss += loss_data.item() * b_len
        total_phys_loss += loss_phys.item() * b_len

    return (
        total_loss / num_samples,
        total_data_loss / num_samples,
        total_phys_loss / num_samples,
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    data_loss_fn: nn.Module,
    phys_loss_fn: nn.Module,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    edge_index: torch.Tensor,
    lambda_phys: float = LAMBDA_PHYSICS,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss, total_data_loss, total_phys_loss = 0.0, 0.0, 0.0
    total_mse = 0.0
    num_samples = X_val.shape[0]

    for start in range(0, num_samples, BATCH_SIZE):
        batch_idx = np.arange(start, min(start + BATCH_SIZE, num_samples))

        x_b = torch.from_numpy(np.asarray(X_val[batch_idx])).float().to(DEVICE)
        y_b = torch.from_numpy(np.asarray(Y_val[batch_idx, :, 0])).float().to(DEVICE)

        mean_pred, log_var_pred = model(x_b, edge_index)

        loss_data = data_loss_fn(mean_pred, log_var_pred, y_b)
        loss_phys = phys_loss_fn(mean_pred, x_b, edge_index)
        loss_total = loss_data + (lambda_phys * loss_phys)

        mse = F.mse_loss(mean_pred, y_b)

        b_len = len(batch_idx)
        total_loss += loss_total.item() * b_len
        total_data_loss += loss_data.item() * b_len
        total_phys_loss += loss_phys.item() * b_len
        total_mse += mse.item() * b_len

    return (
        total_loss / num_samples,
        total_data_loss / num_samples,
        total_phys_loss / num_samples,
        total_mse / num_samples,
    )


# ============================================================
# Main Training & Explainability Loop
# ============================================================

def main():
    print("=" * 70)
    print("PIGT — PHYSICS-INFORMED & UNCERTAINTY-AWARE TRAINING (PERSON 3)")
    print("=" * 70)
    print("Device:", DEVICE)

    X_train, Y_train, X_val, Y_val = load_data()
    edge_index = load_graph()

    # Initialize model
    model = PIGTUncertaintyModel(
        in_features=13,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Initialize loss functions
    data_loss_fn = GaussianNLLLoss()
    phys_loss_fn = OceanAdvectionDiffusionLoss(diffusivity_kappa=DIFFUSIVITY_KAPPA)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "pigt_physics_best.pt"

    print(f"\nTraining for {EPOCHS} epochs (λ_physics = {LAMBDA_PHYSICS})...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_tot, tr_dat, tr_phy = train_one_epoch(
            model, optimizer, data_loss_fn, phys_loss_fn, X_train, Y_train, edge_index
        )
        val_tot, val_dat, val_phy, val_mse = validate(
            model, data_loss_fn, phys_loss_fn, X_val, Y_val, edge_index
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {tr_tot:.4f} (Data: {tr_dat:.4f}, Phys: {tr_phy:.4f}) | "
            f"Val Loss: {val_tot:.4f} (MSE: {val_mse:.4f}) | "
            f"{elapsed:.2f}s"
        )

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_tot,
                    "val_mse": val_mse,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved new best physics-informed model checkpoint: {checkpoint_path.name}")

    print("\n" + "=" * 70)
    print("GENERATING EXPLAINABILITY & UNCERTAINTY ARTIFACTS")
    print("=" * 70)

    # 1. Temporal Attention Analysis
    sample_x = torch.from_numpy(np.asarray(X_val[0:1])).float().to(DEVICE)
    sample_y = np.asarray(Y_val[0, :, 0])

    extractor = TemporalAttentionExtractor(model)
    attn_map = extractor.compute_manual_attention_map(sample_x, edge_index)
    lag_profile = extractor.get_lag_attribution(attn_map)

    plot_temporal_attention_profile(
        lag_profile,
        save_path=FIGURES_DIR / "temporal_attention_profile.png",
    )

    # 2. MC Dropout Uncertainty Estimation
    uq_results = model.predict_with_uncertainty(sample_x, edge_index, num_mc_samples=10)
    pred_mean = uq_results["mean"][0].cpu().numpy()
    pred_std = uq_results["std"][0].cpu().numpy()

    plot_forecast_uncertainty_bounds(
        y_true=sample_y,
        y_pred=pred_mean,
        std=pred_std,
        save_path=FIGURES_DIR / "forecast_uncertainty_bounds.png",
    )

    # 3. Spatial Uncertainty & Residual Maps
    node_lat_p = DATA_DIR / "node_lat.npy"
    node_lon_p = DATA_DIR / "node_lon.npy"
    if node_lat_p.exists() and node_lon_p.exists():
        lats = np.load(node_lat_p)
        lons = np.load(node_lon_p)
    else:
        lats = np.linspace(10, 20, 12204)
        lons = np.linspace(80, 90, 12204)

    plot_spatial_field(
        lats=lats,
        lons=lons,
        values=pred_std,
        title="Bay of Bengal: PIGT Forecast Uncertainty (Standard Deviation)",
        colorbar_label="Predictive Uncertainty σ",
        save_path=FIGURES_DIR / "spatial_uncertainty_map.png",
        cmap="plasma",
    )

    print("\n✓ Person 3 execution and training complete.")


if __name__ == "__main__":
    main()
