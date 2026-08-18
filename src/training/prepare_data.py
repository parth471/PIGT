import numpy as np
import os

# --------------------------------
# Load node features
# Shape:
# (time, nodes, features)
# --------------------------------

features = np.load("data/node_features.npy")

print("Original shape:", features.shape)


# --------------------------------
# Temporal split
# --------------------------------

train = features[:21]
val = features[21:26]
test = features[26:31]

print("\nTemporal split:")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)


# --------------------------------
# Calculate normalization
# ONLY using training data
# --------------------------------

mean = train.mean(axis=(0, 1))
std = train.std(axis=(0, 1))

# Prevent division by zero
std[std == 0] = 1.0

print("\nTraining statistics:")

feature_names = [
    "thetao",
    "so",
    "uo",
    "vo",
    "zos",
    "mlotst"
]

for i, name in enumerate(feature_names):
    print(
        f"{name}: "
        f"mean={mean[i]:.4f}, "
        f"std={std[i]:.4f}"
    )


# --------------------------------
# Normalize
# --------------------------------

train_norm = (train - mean) / std
val_norm = (val - mean) / std
test_norm = (test - mean) / std


# --------------------------------
# Create output directory
# --------------------------------

os.makedirs("data/processed", exist_ok=True)


# --------------------------------
# Save
# --------------------------------

np.save(
    "data/processed/train.npy",
    train_norm
)

np.save(
    "data/processed/val.npy",
    val_norm
)

np.save(
    "data/processed/test.npy",
    test_norm
)

np.save(
    "data/processed/feature_mean.npy",
    mean
)

np.save(
    "data/processed/feature_std.npy",
    std
)


print("\nSaved:")
print("data/processed/train.npy")
print("data/processed/val.npy")
print("data/processed/test.npy")
print("data/processed/feature_mean.npy")
print("data/processed/feature_std.npy")