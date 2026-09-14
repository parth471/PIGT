from pathlib import Path

import numpy as np
import torch

from gcn_baseline import GCNBaseline
from pigt_model import PIGTModel


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    prediction,
    target,
):
    """
    prediction: [samples, nodes]
    target:     [samples, nodes]
    """

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    error = prediction - target

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (target - np.mean(target)) ** 2
    )

    r2 = 1.0 - (
        ss_res / ss_tot
    )

    return mae, rmse, r2


# ============================================================
# Load graph
# ============================================================

def load_graph():

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
    )

    print(
        "Graph nodes:",
        graph["num_nodes"],
    )

    print(
        "Graph edges:",
        graph["edge_index"].shape[1],
    )

    return graph["edge_index"].to(DEVICE)


# ============================================================
# Evaluate GCN
# ============================================================

@torch.no_grad()
def evaluate_gcn(
    model,
    X_test,
    Y_test,
    edge_index,
):

    model.eval()

    predictions = []
    targets = []

    for i in range(
        X_test.shape[0]
    ):

        # GCN uses latest historical timestep
        x = torch.from_numpy(
            np.array(
                X_test[i, -1],
                copy=True,
            )
        ).float().to(DEVICE)

        y = np.array(
            Y_test[i, :, 0],
            copy=True,
        )

        prediction = model(
            x,
            edge_index,
        )

        predictions.append(
            prediction.cpu().numpy()
        )

        targets.append(y)

        print(
            f"\rGCN evaluation: "
            f"{i + 1}/{X_test.shape[0]}",
            end="",
        )

    print()

    predictions = np.stack(
        predictions
    )

    targets = np.stack(
        targets
    )

    return calculate_metrics(
        predictions,
        targets,
    )


# ============================================================
# Evaluate PIGT
# ============================================================

@torch.no_grad()
def evaluate_pigt(
    model,
    X_test,
    Y_test,
    edge_index,
):

    model.eval()

    predictions = []
    targets = []

    for i in range(
        X_test.shape[0]
    ):

        x = torch.from_numpy(
            np.array(
                X_test[i:i + 1],
                copy=True,
            )
        ).float().to(DEVICE)

        y = np.array(
            Y_test[i, :, 0],
            copy=True,
        )

        prediction = model(
            x,
            edge_index,
        )

        predictions.append(
            prediction[0].cpu().numpy()
        )

        targets.append(y)

        print(
            f"\rPIGT evaluation: "
            f"{i + 1}/{X_test.shape[0]}",
            end="",
        )

    print()

    predictions = np.stack(
        predictions
    )

    targets = np.stack(
        targets
    )

    return calculate_metrics(
        predictions,
        targets,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PIGT — TEST SET EVALUATION")
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE,
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test = np.load(
        DATA_DIR / "X_test.npy",
        mmap_mode="r",
    )

    Y_test = np.load(
        DATA_DIR / "Y_test.npy",
        mmap_mode="r",
    )

    print(
        "\nX_test:",
        X_test.shape,
    )

    print(
        "Y_test:",
        Y_test.shape,
    )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    edge_index = load_graph()

    # --------------------------------------------------------
    # GCN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GCN BASELINE")
    print("=" * 70)

    gcn = GCNBaseline(
        in_channels=13,
        hidden_channels=64,
        dropout=0.1,
    ).to(DEVICE)

    gcn_checkpoint = torch.load(
        CHECKPOINT_DIR / "gcn_baseline_best.pt",
        map_location=DEVICE,
    )

    gcn.load_state_dict(
        gcn_checkpoint["model_state_dict"]
    )

    gcn_mae, gcn_rmse, gcn_r2 = evaluate_gcn(
        gcn,
        X_test,
        Y_test,
        edge_index,
    )

    print(
        "\nGCN Test MAE :",
        gcn_mae,
    )

    print(
        "GCN Test RMSE:",
        gcn_rmse,
    )

    print(
        "GCN Test R²  :",
        gcn_r2,
    )

    # --------------------------------------------------------
    # PIGT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIGT")
    print("=" * 70)

    pigt = PIGTModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    pigt_checkpoint = torch.load(
        CHECKPOINT_DIR / "pigt_best.pt",
        map_location=DEVICE,
    )

    pigt.load_state_dict(
        pigt_checkpoint["model_state_dict"]
    )

    pigt_mae, pigt_rmse, pigt_r2 = evaluate_pigt(
        pigt,
        X_test,
        Y_test,
        edge_index,
    )

    print(
        "\nPIGT Test MAE :",
        pigt_mae,
    )

    print(
        "PIGT Test RMSE:",
        pigt_rmse,
    )

    print(
        "PIGT Test R²  :",
        pigt_r2,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<15}"
        f"{'MAE':>15}"
        f"{'RMSE':>15}"
        f"{'R²':>15}"
    )

    print("-" * 60)

    print(
        f"{'GCN':<15}"
        f"{gcn_mae:>15.6f}"
        f"{gcn_rmse:>15.6f}"
        f"{gcn_r2:>15.6f}"
    )

    print(
        f"{'PIGT':<15}"
        f"{pigt_mae:>15.6f}"
        f"{pigt_rmse:>15.6f}"
        f"{pigt_r2:>15.6f}"
    )


if __name__ == "__main__":
    main()