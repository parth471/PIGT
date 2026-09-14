# PIGT — Person 1 Pipeline (Historical Data & Temporal Dataset Preparation)

## Setup

```bash
python -m venv pigt_env
# Windows: pigt_env\Scripts\activate
# macOS/Linux: source pigt_env/bin/activate

pip install -r requirements.txt
```

Then set up credentials (one-time, manual):
- Copernicus Marine: `copernicusmarine login`
- ERA5: create `~/.cdsapirc` with your CDS personal access token (see comments at
  the top of `scripts/download_era5.py`), and accept the ERA5 dataset's terms of
  use on the CDS website once.

## Run order

```bash
cd scripts

# 1. Download raw data (one-time; can take a while, especially ERA5)
python download_copernicus_marine.py
python download_era5.py

# 2. Inspect everything before trusting it
python 01_inspect_raw_data.py

# 3. Align, regrid, merge, mask, gap-fill
python 02_align_and_merge.py

# 4. Normalize, build sequences, chronological split, save final arrays
python 03_build_dataset.py

# 5. Verify the final dataset against the completion checklist
python 04_verify_final_dataset.py
```

## Editing settings

All region/date/window/split settings live in `scripts/config.py`. Change values
there — every script imports from it, so you only edit one file.

## Handoff to Person 2

Everything Person 2 needs is in `data/final/`:

```
X_train.npy, Y_train.npy, X_val.npy, Y_val.npy, X_test.npy, Y_test.npy
lat.npy, lon.npy, node_mask.npy, node_lat.npy, node_lon.npy
feature_scaler.pkl, metadata.json
```

Point them at `metadata.json` first — it documents the region, date range,
history/horizon window, feature order, target order, node count, and how
normalization was done, so they don't have to reverse-engineer any of it.

`X` shape: `[B, T, N, F]` — batch, 7-day history, ocean nodes, 13 features.
`Y` shape: `[B, N, 5]` — batch, ocean nodes, 5 targets
(`sst, salinity, u_current, v_current, swh`, in that fixed order).
