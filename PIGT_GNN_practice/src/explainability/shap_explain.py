import torch
import shap
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

sys.path.insert(0, "src")
sys.path.insert(0, "src/training")

from prepare_data import load_training_graph
from models.gcn_baseline import GCNBaseline


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "data/gcn_baseline_model.pt"

FEATURE_NAMES = [
    "so",
    "zos",
    "uo",
    "vo"
]

# Number of test nodes to explain initially
NUM_NODES = 200

# ---------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------

print("[INFO] Loading graph...")

graph = load_training_graph()

x_original = graph.x.clone()

edge_index = graph.edge_index.clone()

edge_weight = graph.edge_attr.clone()

# Normalize edge weights exactly like training
edge_weight = edge_weight / edge_weight.max()


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print("[INFO] Loading trained model...")

model = GCNBaseline(
    in_dim=x_original.shape[1]
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("[INFO] Model loaded.")


# ---------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------

x_original_device = x_original.to(DEVICE)
edge_index = edge_index.to(DEVICE)
edge_weight = edge_weight.to(DEVICE)

test_mask = graph.test_mask

test_nodes = torch.where(test_mask)[0]

# Select only a small sample initially
test_nodes = test_nodes[:NUM_NODES]

print(
    f"[INFO] Explaining {len(test_nodes)} test nodes..."
)


# ---------------------------------------------------
# PREDICTION WRAPPER
# ---------------------------------------------------

def predict_node(node_idx, feature_values):

    """
    Explain the prediction for one node.

    feature_values:
        [so, zos, uo, vo]

    thetao remains hidden (set to zero).
    """

    x = x_original_device.clone()

    # Hide SST / thetao
    x[:, 0] = 0.0

    # Replace the four input variables
    x[node_idx, 1:] = torch.tensor(
        feature_values,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            x,
            edge_index,
            edge_weight
        )

    return output[node_idx].item()


# ---------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------

all_shap_values = []

print("[INFO] Starting SHAP analysis...")


for count, node_idx in enumerate(test_nodes):

    node_idx = node_idx.item()

    # Original feature values
    original_features = (
        x_original[node_idx, 1:]
        .numpy()
    )

    print(
        f"[INFO] Node {count + 1}/{len(test_nodes)} "
        f"(index={node_idx})"
    )

    # ------------------------------------------------
    # Background
    # ------------------------------------------------

    background = np.array([
        np.zeros(4),
        original_features
    ])

    # ------------------------------------------------
    # SHAP model wrapper
    # ------------------------------------------------

    def shap_model(X):

        predictions = []

        for sample in X:

            pred = predict_node(
                node_idx,
                sample
            )

            predictions.append(pred)

        return np.array(predictions)

    # ------------------------------------------------
    # Kernel SHAP
    # ------------------------------------------------

    explainer = shap.KernelExplainer(
        shap_model,
        background
    )

    shap_values = explainer.shap_values(
        original_features,
        nsamples=50
    )

    shap_values = np.array(shap_values)

    # Handle SHAP output shape
    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    all_shap_values.append(shap_values)


# ---------------------------------------------------
# CONVERT TO ARRAY
# ---------------------------------------------------

all_shap_values = np.array(all_shap_values)

print(
    "[INFO] SHAP array shape:",
    all_shap_values.shape
)


# ---------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# ---------------------------------------------------

mean_abs_shap = np.mean(
    np.abs(all_shap_values),
    axis=0
)

print("\n[RESULTS] Mean absolute SHAP values")

for feature, value in zip(
    FEATURE_NAMES,
    mean_abs_shap
):

    print(
        f"{feature:>5}: {value:.6f}"
    )


# ---------------------------------------------------
# BAR PLOT
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    FEATURE_NAMES,
    mean_abs_shap
)

plt.xlabel("Feature")

plt.ylabel("Mean |SHAP value|")

plt.title(
    "GNN Feature Importance - Bay of Bengal"
)

plt.tight_layout()


os.makedirs(
    "data/shap",
    exist_ok=True
)

output_path = (
    "data/shap/"
    "feature_importance.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()

print(
    f"[INFO] Saved plot to {output_path}"
)