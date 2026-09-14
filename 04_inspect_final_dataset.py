from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "data" / "final"


def inspect_npy(filename):
    path = FINAL_DIR / filename

    if not path.exists():
        print(f"\n[MISSING] {filename}")
        return

    data = np.load(path)

    print(f"\n--- {filename} ---")
    print("Shape :", data.shape)
    print("Dtype :", data.dtype)
    print("Min   :", np.min(data))
    print("Max   :", np.max(data))
    print("Mean  :", np.mean(data))
    print("Std   :", np.std(data))
    print("NaNs  :", np.isnan(data).sum())
    print("Infs  :", np.isinf(data).sum())


def main():

    print("=" * 60)
    print("PIGT FINAL DATASET INSPECTION")
    print("=" * 60)

    files = [
        "X_train.npy",
        "Y_train.npy",
        "X_val.npy",
        "Y_val.npy",
        "X_test.npy",
        "Y_test.npy",
        "node_mask.npy",
        "node_lat.npy",
        "node_lon.npy",
        "feature_scaler.pkl",
        "metadata.json",
    ]

    print("\nFILE CHECK")
    print("-" * 60)

    for filename in files:
        path = FINAL_DIR / filename
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{filename:20} : {status}")

    metadata_path = FINAL_DIR / "metadata.json"

    if metadata_path.exists():
        print("\n" + "=" * 60)
        print("METADATA")
        print("=" * 60)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(json.dumps(metadata, indent=2))

    for filename in [
        "X_train.npy",
        "Y_train.npy",
        "X_val.npy",
        "Y_val.npy",
        "X_test.npy",
        "Y_test.npy",
        "node_mask.npy",
        "node_lat.npy",
        "node_lon.npy",
    ]:
        inspect_npy(filename)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()