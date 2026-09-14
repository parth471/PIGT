from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

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

EPOCHS = 1
BATCH_SIZE = 1

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


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    X_train,
    Y_train,
    edge_index,
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

        # SST target only = target channel 0
        y_batch = torch.from_numpy(
            np.asarray(
                Y_train[batch_indices, :, 0]
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

        loss = F.mse_loss(
            prediction,
            y_batch,
        )

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
            np.asarray(
                Y_val[
                    batch_indices,
                    :,
                    0,
                ]
            )
        ).float()

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        prediction = model(
            x_batch,
            edge_index,
        )

        loss = F.mse_loss(
            prediction,
            y_batch,
        )

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
    print("PIGT — GRAPH + TEMPORAL TRANSFORMER TRAINING")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    X_train, Y_train, X_val, Y_val = load_data()

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
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "pigt_best.pt"
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "=" * 70)
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print("=" * 70)

        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            optimizer,
            X_train,
            Y_train,
            edge_index,
        )

        val_loss = validate(
            model,
            X_val,
            Y_val,
            edge_index,
        )

        elapsed = time.time() - start_time

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
                    },
                },
                checkpoint_path,
            )

            print(
                "\n✓ New best model saved:"
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

    print(
        "Best checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()