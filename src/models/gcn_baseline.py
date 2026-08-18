import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ============================================================
# 1. Load processed data
# ============================================================

train = np.load("data/processed/train.npy")
val = np.load("data/processed/val.npy")
test = np.load("data/processed/test.npy")

edges = pd.read_csv("data/edges.csv")

# Feature statistics from training data
mean = np.load("data/processed/feature_mean.npy")
std = np.load("data/processed/feature_std.npy")

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)


# ============================================================
# 2. Build graph structure
# ============================================================

edge_index = torch.tensor(
    edges[["source", "target"]].values.T,
    dtype=torch.long
)

edge_distance = torch.tensor(
    edges["distance_km"].values,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# 3. Create PyG graph for one timestep
# ============================================================

def create_graph(features, timestep):
    x = torch.tensor(
        features[timestep],
        dtype=torch.float32
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_distance
    )


# ============================================================
# 4. GCN model
# ============================================================

class GCNBaseline(nn.Module):

    def __init__(self, input_features=6, hidden_features=32):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features
        )

        self.conv2 = GCNConv(
            hidden_features,
            hidden_features
        )

        self.output = nn.Linear(
            hidden_features,
            1
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # Predict next-day normalized SST
        x = self.output(x)

        return x.squeeze(-1)


# ============================================================
# 5. Model + optimizer
# ============================================================

device = torch.device("cpu")

model = GCNBaseline().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_function = nn.MSELoss()


# ============================================================
# 6. Training
# ============================================================

print("\nStarting training...\n")

for epoch in range(1, 21):

    model.train()

    total_loss = 0.0

    # Train on Day 1 → Day 2 ... Day 20 → Day 21
    for t in range(len(train) - 1):

        graph = create_graph(train, t)

        target = torch.tensor(
            train[t + 1, :, 0],
            dtype=torch.float32
        )

        graph = graph.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        prediction = model(
            graph.x,
            graph.edge_index
        )

        loss = loss_function(
            prediction,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / (len(train) - 1)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    validation_losses = []

    with torch.no_grad():

        for t in range(len(val) - 1):

            graph = create_graph(val, t)

            target = torch.tensor(
                val[t + 1, :, 0],
                dtype=torch.float32
            )

            prediction = model(
                graph.x,
                graph.edge_index
            )

            val_loss = loss_function(
                prediction,
                target
            )

            validation_losses.append(
                val_loss.item()
            )

    validation_loss = np.mean(
        validation_losses
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {average_loss:.6f} | "
        f"Val Loss: {validation_loss:.6f}"
    )


# ============================================================
# 7. Test
# ============================================================

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for t in range(len(test) - 1):

        graph = create_graph(test, t)

        target = test[t + 1, :, 0]

        prediction = model(
            graph.x,
            graph.edge_index
        )

        predictions.append(
            prediction.numpy()
        )

        actual.append(target)


predictions = np.concatenate(predictions)
actual = np.concatenate(actual)


# ============================================================
# 8. Convert normalized SST back to °C
# ============================================================

thetao_mean = mean[0]
thetao_std = std[0]

predictions_c = (
    predictions * thetao_std
    + thetao_mean
)

actual_c = (
    actual * thetao_std
    + thetao_mean
)


# ============================================================
# 9. Metrics
# ============================================================

mae = np.mean(
    np.abs(predictions_c - actual_c)
)

rmse = np.sqrt(
    np.mean(
        (predictions_c - actual_c) ** 2
    )
)

print("\n==============================")
print("GCN TEST RESULTS")
print("==============================")

print(f"MAE:  {mae:.4f} °C")
print(f"RMSE: {rmse:.4f} °C")


# ============================================================
# 10. Persistence baseline
# ============================================================

# Simplest possible prediction:
# tomorrow's SST = today's SST

persistence_predictions = []

persistence_actual = []

for t in range(len(test) - 1):

    persistence_predictions.extend(
        test[t, :, 0]
    )

    persistence_actual.extend(
        test[t + 1, :, 0]
    )

persistence_predictions = np.array(
    persistence_predictions
)

persistence_actual = np.array(
    persistence_actual
)

persistence_predictions_c = (
    persistence_predictions * thetao_std
    + thetao_mean
)

persistence_actual_c = (
    persistence_actual * thetao_std
    + thetao_mean
)

persistence_mae = np.mean(
    np.abs(
        persistence_predictions_c
        - persistence_actual_c
    )
)

persistence_rmse = np.sqrt(
    np.mean(
        (
            persistence_predictions_c
            - persistence_actual_c
        ) ** 2
    )
)

print("\n==============================")
print("PERSISTENCE BASELINE")
print("==============================")

print(
    f"MAE:  {persistence_mae:.4f} °C"
)

print(
    f"RMSE: {persistence_rmse:.4f} °C"
)