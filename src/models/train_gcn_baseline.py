from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from gcn_baseline import GCNBaseline


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
# Configuration
# ============================================================

EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

HIDDEN_DIM = 64
DROPOUT = 0.1

PATIENCE = 3
GRAD_CLIP = 1.0


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load data
# ============================================================

def load_data():

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

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
    )

    edge_index = graph["edge_index"]

    print("Graph nodes:", graph["num_nodes"])
    print("Graph edges:", edge_index.shape[1])

    return edge_index.to(DEVICE)


# ============================================================
# Train
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

    # Each training sample is one 7-day sequence.
    # For the GCN baseline we only use the latest day.
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    for sample_idx in indices:

        x = torch.from_numpy(
            np.array(
                X_train[sample_idx, -1],
                copy=True,
            )
        ).float().to(DEVICE)

        y = torch.from_numpy(
            np.array(
                Y_train[sample_idx, :, 0],
                copy=True,
            )
        ).float().to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            x,
            edge_index,
        )

        loss = F.mse_loss(
            prediction,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += loss.item()

        print(
            f"\rTraining: "
            f"{sample_idx + 1}",
            end="",
        )

    print()

    return total_loss / X_train.shape[0]


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

    for sample_idx in range(
        X_val.shape[0]
    ):

        x = torch.from_numpy(
            np.asarray(
                X_val[sample_idx, -1]
            )
        ).float().to(DEVICE)

        y = torch.from_numpy(
            np.asarray(
                Y_val[sample_idx, :, 0]
            )
        ).float().to(DEVICE)

        prediction = model(
            x,
            edge_index,
        )

        loss = F.mse_loss(
            prediction,
            y,
        )

        total_loss += loss.item()

    return total_loss / X_val.shape[0]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PIGT — GCN BASELINE TRAINING")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    X_train, Y_train, X_val, Y_val = load_data()

    edge_index = load_graph()

    model = GCNBaseline(
        in_channels=13,
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "gcn_baseline_best.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
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
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Val loss:   {val_loss:.6f}"
        )

        print(
            f"Time:       {elapsed:.2f} sec"
        )

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
                },
                checkpoint_path,
            )

            print(
                "\n✓ Best GCN checkpoint saved:"
            )
            print(checkpoint_path)

        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation improvement."
            )

            print(
                f"Early stopping: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\n" + "=" * 70)
    print("GCN BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()