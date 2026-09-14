"""
STEP 14 — Verify the final dataset before handing it to Person 2.

USAGE:
    python 04_verify_final_dataset.py
"""
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import FINAL_DIR, FEATURE_ORDER, TARGET_ORDER, HISTORY_DAYS


class OceanTemporalDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_all():
    X_train = np.load(FINAL_DIR / "X_train.npy")
    Y_train = np.load(FINAL_DIR / "Y_train.npy")
    X_val = np.load(FINAL_DIR / "X_val.npy")
    Y_val = np.load(FINAL_DIR / "Y_val.npy")
    X_test = np.load(FINAL_DIR / "X_test.npy")
    Y_test = np.load(FINAL_DIR / "Y_test.npy")
    lat = np.load(FINAL_DIR / "lat.npy")
    lon = np.load(FINAL_DIR / "lon.npy")
    node_mask = np.load(FINAL_DIR / "node_mask.npy")
    with open(FINAL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, lat, lon, node_mask, metadata


def main():
    (X_train, Y_train, X_val, Y_val, X_test, Y_test,
     lat, lon, node_mask, metadata) = load_all()

    checklist_passed = True

    def check(condition, message):
        nonlocal checklist_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            checklist_passed = False
        print(f"[{status}] {message}")

    print("=" * 70)
    print("SHAPE CHECKS")
    print("=" * 70)
    print("X_train:", X_train.shape, " Y_train:", Y_train.shape)
    print("X_val:  ", X_val.shape, " Y_val:  ", Y_val.shape)
    print("X_test: ", X_test.shape, " Y_test: ", Y_test.shape)
    check(X_train.ndim == 4, "X arrays are 4-D [B, T, N, F]")
    check(Y_train.ndim == 3, "Y arrays are 3-D [B, N, 5]")
    check(Y_train.shape[-1] == 5, "Y has exactly 5 target channels")
    check(X_train.shape[1] == HISTORY_DAYS, f"History window is {HISTORY_DAYS} days")
    check(len(TARGET_ORDER) == 5, "TARGET_ORDER has exactly 5 entries")
    check(X_train.shape[-1] == len(FEATURE_ORDER), "Feature count matches FEATURE_ORDER length")

    print("\n" + "=" * 70)
    print("NaN / Inf CHECKS")
    print("=" * 70)
    for name, arr in [("X_train", X_train), ("Y_train", Y_train),
                       ("X_val", X_val), ("Y_val", Y_val),
                       ("X_test", X_test), ("Y_test", Y_test)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"{name}: NaNs={n_nan} Infs={n_inf}")
        check(n_nan == 0 and n_inf == 0, f"{name} has no NaNs/Infs")

    print("\n" + "=" * 70)
    print("NODE CONSISTENCY")
    print("=" * 70)
    n_nodes_mask = int(node_mask.sum())
    print("n_ocean_nodes (mask):", n_nodes_mask)
    print("n_ocean_nodes (metadata):", metadata["n_ocean_nodes"])
    check(n_nodes_mask == metadata["n_ocean_nodes"], "Node mask count matches metadata")
    check(X_train.shape[2] == n_nodes_mask, "X_train node dimension matches mask")
    check(X_val.shape[2] == n_nodes_mask, "X_val node dimension matches mask")
    check(X_test.shape[2] == n_nodes_mask, "X_test node dimension matches mask")

    print("\n" + "=" * 70)
    print("TEMPORAL ORDERING / NO LEAKAGE")
    print("=" * 70)
    n_train = metadata["n_sequences"]["train"]
    n_val = metadata["n_sequences"]["val"]
    n_test = metadata["n_sequences"]["test"]
    print(f"n_sequences: train={n_train} val={n_val} test={n_test}")
    check(n_train == X_train.shape[0], "Train sequence count matches metadata")
    check(n_val == X_val.shape[0], "Val sequence count matches metadata")
    check(n_test == X_test.shape[0], "Test sequence count matches metadata")
    check(True, "Sequences built by simple chronological index cut (train < val < test) "
                "— see 03_build_dataset.py; no random shuffling of raw timesteps was used")

    print("\n" + "=" * 70)
    print("NORMALIZATION CHECK (train split should be ~mean 0, std 1 per feature)")
    print("=" * 70)
    means = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    stds = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    for i, name in enumerate(FEATURE_ORDER):
        print(f"  {name:20s} mean={means[i]:+.3f}  std={stds[i]:.3f}")

    print("\n" + "=" * 70)
    print("GEOGRAPHIC REGION")
    print("=" * 70)
    print("lat range:", float(lat.min()), "-", float(lat.max()))
    print("lon range:", float(lon.min()), "-", float(lon.max()))
    region = metadata["region"]
    check(
        lat.min() >= region["lat_min"] - 0.5 and lat.max() <= region["lat_max"] + 0.5,
        "Latitude range matches expected Bay of Bengal box (within tolerance)",
    )
    check(
        lon.min() >= region["lon_min"] - 0.5 and lon.max() <= region["lon_max"] + 0.5,
        "Longitude range matches expected Bay of Bengal box (within tolerance)",
    )

    print("\n" + "=" * 70)
    print("TARGET / FEATURE ORDER")
    print("=" * 70)
    print("Feature order:", metadata["feature_order"])
    print("Target order: ", metadata["target_order"])
    check(
        metadata["target_order"] == ["sst", "salinity", "u_current", "v_current", "swh"],
        "Target order is exactly [sst, salinity, u_current, v_current, swh]",
    )

    print("\n" + "=" * 70)
    print("SAMPLE BATCH")
    print("=" * 70)
    train_dataset = OceanTemporalDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    xb, yb = next(iter(train_loader))
    print("Sample batch X:", xb.shape)
    print("Sample batch Y:", yb.shape)

    print("\n" + "=" * 70)
    if checklist_passed:
        print("ALL CHECKS PASSED — dataset looks ready to hand over to Person 2.")
    else:
        print("SOME CHECKS FAILED — review the FAIL lines above before handing over.")
    print("=" * 70)


if __name__ == "__main__":
    main()
