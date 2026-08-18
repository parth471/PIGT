import numpy as np

features = np.load("data/node_features.npy")

feature_names = [
    "thetao",
    "so",
    "uo",
    "vo",
    "zos",
    "mlotst"
]

print("Feature shape:", features.shape)
print()

for i, name in enumerate(feature_names):

    values = features[:, :, i]

    print(f"{name}:")
    print(f"  min  = {values.min():.4f}")
    print(f"  max  = {values.max():.4f}")
    print(f"  mean = {values.mean():.4f}")
    print(f"  std  = {values.std():.4f}")
    print()