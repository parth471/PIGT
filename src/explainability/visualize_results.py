from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


def plot_temporal_attention_profile(
    lag_weights: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "PIGT 7-Day Temporal Attention Profile",
) -> None:
    """
    Plots the attention importance distribution over the 7-day historical window.
    """
    t_steps = len(lag_weights)
    days_labels = [f"t-{t_steps - 1 - i}" if i < t_steps - 1 else "t (Today)" for i in range(t_steps)]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(days_labels, lag_weights, color="#1f77b4", edgecolor="#0d3d63", alpha=0.85, width=0.6)

    # Highlight today's lag
    bars[-1].set_color("#d62728")
    bars[-1].set_edgecolor("#6b1414")

    ax.set_xlabel("Historical Lag (Days)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attention Weight", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✓ Saved attention plot to: {save_path}")
    plt.close()


def plot_spatial_field(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    save_path: Optional[Path] = None,
    cmap: str = "viridis",
) -> None:
    """
    Plots a geographic scatter/contour heatmap over the Bay of Bengal ocean nodes.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    scatter = ax.scatter(
        lons,
        lats,
        c=values,
        cmap=cmap,
        s=4,
        alpha=0.9,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✓ Saved spatial plot to: {save_path}")
    plt.close()


def plot_forecast_uncertainty_bounds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std: np.ndarray,
    save_path: Optional[Path] = None,
    num_nodes_sample: int = 50,
) -> None:
    """
    Plots predicted SST with ±2σ (95%) confidence intervals against true targets.
    """
    indices = np.arange(min(num_nodes_sample, len(y_true)))

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    ax.plot(indices, y_true[indices], "o-", label="Ground Truth SST", color="#2ca02c", markersize=4)
    ax.plot(indices, y_pred[indices], "s--", label="PIGT Predicted Mean (μ)", color="#1f77b4", markersize=4)

    # 95% Confidence bounds (± 2 * std)
    ax.fill_between(
        indices,
        y_pred[indices] - 2 * std[indices],
        y_pred[indices] + 2 * std[indices],
        color="#1f77b4",
        alpha=0.2,
        label="95% Uncertainty Bound (±2σ)",
    )

    ax.set_title("SST Forecast with Calibrated Uncertainty Bounds", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Ocean Node Index", fontsize=10, fontweight="bold")
    ax.set_ylabel("SST (Normalized)", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✓ Saved forecast uncertainty plot to: {save_path}")
    plt.close()
