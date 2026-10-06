from pathlib import Path
import os
import json
import pickle
import time

import numpy as np
import torch

from pigt_model import PIGTModel


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final"
GRAPH_PATH = ROOT / "data" / "graph" / "graph.pt"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Training configuration
# ============================================================

MAX_EPOCHS = int(os.environ.get("PIGT_MAX_EPOCHS", "50"))
MAX_TRAIN_SAMPLES = os.environ.get("PIGT_MAX_TRAIN_SAMPLES")
MAX_TRAIN_SAMPLES = int(MAX_TRAIN_SAMPLES) if MAX_TRAIN_SAMPLES else None
CHECKPOINT_NAME = os.environ.get("PIGT_CHECKPOINT_NAME", "pigt_8target_best.pt")
RUN_LABEL = os.environ.get("PIGT_RUN_LABEL", "FULL TRAINING")
BATCH_SIZE = 1
TARGET_ORDER = [
    "sst", "salinity", "u_current", "v_current", "swh",
    "wind_speed", "wind_direction", "pressure",
]

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

PATIENCE = 5

GRAD_CLIP = 1.0


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load data
# ============================================================

def load_data():

    print("\nLoading datasets...")

    # Memory mapping prevents the entire large training array
    # from being loaded into RAM.
    X_train = np.load(
        DATA_DIR / "X_train.npy",
        mmap_mode="r",
    )

    Y_train = np.load(
        DATA_DIR / "Y_train.npy",
        mmap_mode="r",
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy",
        mmap_mode="r",
    )

    Y_val = np.load(
        DATA_DIR / "Y_val.npy",
        mmap_mode="r",
    )

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape)
    print("Y_val  :", Y_val.shape)

    return X_train, Y_train, X_val, Y_val


# ============================================================
# Load graph
# ============================================================

def load_graph():

    print("\nLoading graph...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
    )

    edge_index = graph["edge_index"]

    print(
        "Graph nodes:",
        graph["num_nodes"],
    )

    print(
        "Graph edges:",
        edge_index.shape[1],
    )

    return edge_index.to(DEVICE)


# ============================================================
# Create model
# ============================================================

def create_model():

    model = PIGTModel(
        in_features=13,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        dropout=DROPOUT,
    )

    return model.to(DEVICE)


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

    direction_scaler = target_scalers["wind_direction"]
    if not hasattr(direction_scaler, "mean_") or not hasattr(direction_scaler, "scale_"):
        raise TypeError("wind_direction scaler must expose mean_ and scale_.")

    return target_scalers


def eight_target_loss(prediction, target, target_scalers):
    """Equal-weight standardized MSE with wrapped direction error."""
    if prediction.shape != target.shape or prediction.shape[-1] != len(TARGET_ORDER):
        raise ValueError(
            f"Expected matching [B, N, 8] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    channel_losses = []
    for channel, name in enumerate(TARGET_ORDER):
        if name == "wind_direction":
            scaler = target_scalers[name]
            mean = prediction.new_tensor(float(scaler.mean_[0]))
            scale = prediction.new_tensor(float(scaler.scale_[0]))
            pred_degrees = prediction[..., channel] * scale + mean
            target_degrees = target[..., channel] * scale + mean
            delta_radians = torch.deg2rad(pred_degrees - target_degrees)
            wrapped_radians = torch.atan2(
                torch.sin(delta_radians), torch.cos(delta_radians)
            )
            wrapped_standardized_error = torch.rad2deg(wrapped_radians) / scale
            channel_losses.append(wrapped_standardized_error.square().mean())
        else:
            channel_losses.append(
                (prediction[..., channel] - target[..., channel]).square().mean()
            )

    return torch.stack(channel_losses).mean()


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    X_train,
    Y_train,
    edge_index,
    target_scalers,
):

    model.train()

    total_loss = 0.0
    num_samples = X_train.shape[0]

    indices = np.arange(num_samples)

    # Shuffle training samples.
    # This does NOT shuffle the temporal structure inside each
    # 7-day sequence and does not affect the chronological split.
    np.random.shuffle(indices)

    for start in range(
        0,
        num_samples,
        BATCH_SIZE,
    ):

        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        x_batch = torch.from_numpy(
            np.asarray(
                X_train[batch_indices]
            )
        ).float()

        y_batch = torch.from_numpy(
            np.asarray(
                Y_train[batch_indices]
            )
        ).float()

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            x_batch,
            edge_index,
        )

        # ----------------------------------------------------
        # Data loss
        # ----------------------------------------------------

        loss = eight_target_loss(prediction, y_batch, target_scalers)

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * len(batch_indices)
        )

        print(
            f"\rTraining: "
            f"{min(start + BATCH_SIZE, num_samples)}/"
            f"{num_samples}",
            end="",
        )

    print()

    return total_loss / num_samples


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    X_val,
    Y_val,
    edge_index,
    target_scalers,
):

    model.eval()

    total_loss = 0.0
    num_samples = X_val.shape[0]

    for start in range(
        0,
        num_samples,
        BATCH_SIZE,
    ):

        batch_indices = np.arange(
            start,
            min(
                start + BATCH_SIZE,
                num_samples,
            ),
        )

        x_batch = torch.from_numpy(
            np.asarray(
                X_val[batch_indices]
            )
        ).float()

        y_batch = torch.from_numpy(
            np.asarray(Y_val[batch_indices])
        ).float()

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        prediction = model(
            x_batch,
            edge_index,
        )

        loss = eight_target_loss(prediction, y_batch, target_scalers)

        total_loss += (
            loss.item()
            * len(batch_indices)
        )

    return total_loss / num_samples


# ============================================================
# Main training loop
# ============================================================

def main():

    print("=" * 70)
    print(f"PIGT — {RUN_LABEL}")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    X_train, Y_train, X_val, Y_val = load_data()
    target_scalers = load_target_scalers()

    if MAX_TRAIN_SAMPLES is not None:
        if MAX_TRAIN_SAMPLES < 1:
            raise ValueError("PIGT_MAX_TRAIN_SAMPLES must be positive.")
        X_train = X_train[:MAX_TRAIN_SAMPLES]
        Y_train = Y_train[:MAX_TRAIN_SAMPLES]
        print(f"Training subset: first {len(X_train)} training sequences")

    expected_x_tail = (7, 12204, 13)
    expected_y_tail = (12204, 8)
    if X_train.shape[1:] != expected_x_tail or X_val.shape[1:] != expected_x_tail:
        raise ValueError(
            f"Expected X samples shaped [7, 12204, 13], got "
            f"{X_train.shape} and {X_val.shape}"
        )
    if Y_train.shape[1:] != expected_y_tail or Y_val.shape[1:] != expected_y_tail:
        raise ValueError(
            f"Expected Y samples shaped [12204, 8], got "
            f"{Y_train.shape} and {Y_val.shape}"
        )

    edge_index = load_graph()

    model = create_model()

    print("\nModel:")
    print(model)

    print(
        "\nTrainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_val_epoch = 0
    epoch_history = []
    epochs_without_improvement = 0

    if Path(CHECKPOINT_NAME).name != CHECKPOINT_NAME:
        raise ValueError("PIGT_CHECKPOINT_NAME must be a filename, not a path.")
    if CHECKPOINT_NAME in {"pigt_best.pt", "pigt_8target_best.pt"}:
        raise ValueError("Refusing to overwrite an existing full-training checkpoint.")
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        print("\n" + "=" * 70)
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}"
        )
        print("=" * 70)

        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            optimizer,
            X_train,
            Y_train,
            edge_index,
            target_scalers,
        )

        val_loss = validate(
            model,
            X_val,
            Y_val,
            edge_index,
            target_scalers,
        )

        elapsed = time.time() - start_time
        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": elapsed,
        })

        print(
            f"\nTrain loss: {train_loss:.6f}"
        )

        print(
            f"Val loss:   {val_loss:.6f}"
        )

        print(
            f"Time:       {elapsed:.2f} sec"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": {
                        "hidden_dim": HIDDEN_DIM,
                        "num_heads": NUM_HEADS,
                        "num_transformer_layers": NUM_TRANSFORMER_LAYERS,
                        "dropout": DROPOUT,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        "max_epochs": MAX_EPOCHS,
                        "target_order": TARGET_ORDER,
                        "training_samples": int(X_train.shape[0]),
                        "run_label": RUN_LABEL,
                    },
                },
                checkpoint_path,
            )

            print(
                "\n[OK] New best model saved:"
            )

            print(
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation improvement."
            )

            print(
                "Early stopping counter:",
                f"{epochs_without_improvement}/{PATIENCE}",
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print("Best validation epoch:", best_val_epoch)
    if epoch_history:
        improved_after_epoch_one = min(
            (entry["val_loss"] for entry in epoch_history[1:]),
            default=float("inf"),
        ) < epoch_history[0]["val_loss"]
        print("Validation improved over epoch 1:", improved_after_epoch_one)

    print(
        "Best checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
