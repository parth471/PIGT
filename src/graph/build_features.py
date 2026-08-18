import xarray as xr
import pandas as pd
import numpy as np

# -----------------------------
# Load data
# -----------------------------
ds = xr.open_dataset(
    "data/raw/arabian_sea_physics.nc"
)

nodes = pd.read_csv(
    "data/processed_nodes.csv"
)

# -----------------------------
# Variables
# -----------------------------
variables = [
    "thetao",
    "so",
    "uo",
    "vo",
    "zos",
    "mlotst"
]

# -----------------------------
# Extract features
# -----------------------------

feature_arrays = []

for variable in variables:

    data = ds[variable]

    # thetao, so, uo, vo have depth dimension
    if "depth" in data.dims:
        data = data.isel(depth=0)

    # Convert to numpy
    values = data.values

    # Select only valid ocean nodes
    node_values = values[
        :,
        nodes["lat_index"].values,
        nodes["lon_index"].values
    ]

    feature_arrays.append(node_values)


# -----------------------------
# Stack variables
# -----------------------------
features = np.stack(
    feature_arrays,
    axis=-1
)

print("Feature shape:")
print(features.shape)

print("\nExpected:")
print("(time, nodes, features)")

# -----------------------------
# Check missing values
# -----------------------------
print("\nMissing values:")
print(np.isnan(features).sum())

# -----------------------------
# Save
# -----------------------------
np.save(
    "data/node_features.npy",
    features
)

print("\nSaved:")
print("data/node_features.npy")