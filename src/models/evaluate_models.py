import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from pigt_model import PIGTModel


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"
CHECKPOINT_PATH = ROOT / "results" / "checkpoints" / os.environ.get(
    "PIGT_CHECKPOINT_NAME", "pigt_8target_best.pt"
)
REPORT_PATH = ROOT / os.environ.get(
    "PIGT_REPORT_PATH", "results/evaluation/pigt_8target_test_metrics.json"
)
RUN_LABEL = os.environ.get("PIGT_RUN_LABEL", "PIGT EIGHT-TARGET EVALUATION")
TARGET_ORDER = [
    "sst", "salinity", "u_current", "v_current", "swh",
    "wind_speed", "wind_direction", "pressure",
]
LINEAR_TARGETS = [name for name in TARGET_ORDER if name != "wind_direction"]
TARGET_UNITS = {
    "sst": "degrees_C",
    "salinity": "1e-3",
    "u_current": "m s-1",
    "v_current": "m s-1",
    "swh": "m",
    "wind_speed": "m s-1",
    "wind_direction": "degrees",
    "pressure": "hPa",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_target_scalers():
    with open(DATA_DIR / "feature_scaler.pkl", "rb") as f:
        scaler_bundle = pickle.load(f)
    target_scalers = scaler_bundle["targets"]
    if list(target_scalers) != TARGET_ORDER:
        raise ValueError(
            "Target scaler order does not match the required target order: "
            f"{list(target_scalers)}"
        )
    with open(DATA_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("target_order") != TARGET_ORDER:
        raise ValueError(
            "Dataset metadata target order does not match the required order: "
            f"{metadata.get('target_order')}"
        )
    return target_scalers


def inverse_target_channel(values, scaler):
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


@torch.no_grad()
def evaluate_pigt(model, X_test, Y_test, edge_index, target_scalers):
    model.eval()
    linear_totals = {
        name: {key: 0.0 for key in ("abs_error", "squared_error", "target_sum", "target_squared_sum", "count")}
        for name in LINEAR_TARGETS
    }
    direction_totals = {key: 0.0 for key in ("abs_error_degrees", "squared_error_degrees", "abs_error_standardized", "squared_error_standardized", "count")}
    overall_standardized_abs = 0.0
    overall_standardized_squared = 0.0
    overall_standardized_count = 0

    for sample_index in range(X_test.shape[0]):
        x = torch.from_numpy(
            np.array(X_test[sample_index:sample_index + 1], copy=True)
        ).float().to(DEVICE)
        prediction = model(x, edge_index)[0].cpu().numpy()
        target = np.array(Y_test[sample_index], copy=True)

        if prediction.shape != target.shape or prediction.shape != (12204, 8):
            raise ValueError(
                f"Expected prediction and target shape [12204, 8], got "
                f"{prediction.shape} and {target.shape}"
            )

        for channel, name in enumerate(TARGET_ORDER):
            pred_scaled = prediction[:, channel]
            target_scaled = target[:, channel]
            if name == "wind_direction":
                scaler = target_scalers[name]
                pred_degrees = inverse_target_channel(pred_scaled, scaler)
                target_degrees = inverse_target_channel(target_scaled, scaler)
                delta_degrees = (pred_degrees - target_degrees + 180.0) % 360.0 - 180.0
                standardized_delta = delta_degrees / float(scaler.scale_[0])

                direction_totals["abs_error_degrees"] += float(np.abs(delta_degrees).sum())
                direction_totals["squared_error_degrees"] += float(np.square(delta_degrees).sum())
                direction_totals["abs_error_standardized"] += float(np.abs(standardized_delta).sum())
                direction_totals["squared_error_standardized"] += float(np.square(standardized_delta).sum())
                direction_totals["count"] += delta_degrees.size
                overall_standardized_abs += float(np.abs(standardized_delta).sum())
                overall_standardized_squared += float(np.square(standardized_delta).sum())
                overall_standardized_count += standardized_delta.size
            else:
                scaler = target_scalers[name]
                pred_physical = inverse_target_channel(pred_scaled, scaler)
                target_physical = inverse_target_channel(target_scaled, scaler)
                error_physical = pred_physical - target_physical
                total = linear_totals[name]
                total["abs_error"] += float(np.abs(error_physical).sum())
                total["squared_error"] += float(np.square(error_physical).sum())
                total["target_sum"] += float(target_physical.sum())
                total["target_squared_sum"] += float(np.square(target_physical).sum())
                total["count"] += error_physical.size

                error_standardized = pred_scaled - target_scaled
                overall_standardized_abs += float(np.abs(error_standardized).sum())
                overall_standardized_squared += float(np.square(error_standardized).sum())
                overall_standardized_count += error_standardized.size

        print(f"\rPIGT test evaluation: {sample_index + 1}/{X_test.shape[0]}", end="")
    print()

    metrics = {}
    for name in TARGET_ORDER:
        if name == "wind_direction":
            count = direction_totals["count"]
            metrics[name] = {
                "circular_mae_degrees": direction_totals["abs_error_degrees"] / count,
                "circular_rmse_degrees": float(np.sqrt(direction_totals["squared_error_degrees"] / count)),
                "r2": None,
                "r2_note": "Omitted: ordinary R² is not appropriate for a circular angle target.",
            }
        else:
            total = linear_totals[name]
            count = total["count"]
            target_ss = total["target_squared_sum"] - total["target_sum"] ** 2 / count
            r2 = None if target_ss <= 0 else 1.0 - total["squared_error"] / target_ss
            metrics[name] = {
                "mae": total["abs_error"] / count,
                "rmse": float(np.sqrt(total["squared_error"] / count)),
                "r2": r2,
                "units": TARGET_UNITS[name],
            }

    overall = {
        "standardized_mae": overall_standardized_abs / overall_standardized_count,
        "standardized_rmse": float(np.sqrt(overall_standardized_squared / overall_standardized_count)),
        "aggregation": (
            "Pooled equally over all target values: ordinary standardized error for seven linear targets; "
            "shortest wrapped angular error divided by the wind_direction scaler scale for direction."
        ),
    }
    return metrics, overall


def main():
    print("=" * 70)
    print(RUN_LABEL)
    print("=" * 70)
    print("Device:", DEVICE)

    X_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    Y_test = np.load(DATA_DIR / "Y_test.npy", mmap_mode="r")
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)
    if X_test.ndim != 4 or X_test.shape[1:] != (7, 12204, 13):
        raise ValueError(f"Unexpected X_test shape: {X_test.shape}")
    if Y_test.ndim != 3 or Y_test.shape[1:] != (12204, 8):
        raise ValueError(f"Unexpected Y_test shape: {Y_test.shape}")

    target_scalers = load_target_scalers()
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=True)
    if graph["num_nodes"] != 12204:
        raise ValueError(f"Expected 12204 graph nodes, got {graph['num_nodes']}")
    edge_index = graph["edge_index"].to(DEVICE)
    print("Graph nodes:", graph["num_nodes"], "edges:", edge_index.shape[1])

    model = PIGTModel(
        in_features=13,
        hidden_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.1,
    ).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    per_target, overall = evaluate_pigt(
        model, X_test, Y_test, edge_index, target_scalers
    )
    report = {
        "run_label": RUN_LABEL,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "split": "test",
        "n_samples": int(X_test.shape[0]),
        "target_order": TARGET_ORDER,
        "input_shape": ["B", 7, 12204, 13],
        "output_shape": ["B", 12204, 8],
        "per_target_metrics": per_target,
        "overall_standardized_error": overall,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(json.dumps(report, indent=2))
    print("Saved evaluation report:", REPORT_PATH)


if __name__ == "__main__":
    main()
