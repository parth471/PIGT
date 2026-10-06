# PIGT — Bay of Bengal Ocean Forecasting

## 1. Project Overview

This project develops a **multivariate spatiotemporal ocean forecasting system for the Bay of Bengal** using oceanographic, atmospheric, and spatial data.

The project uses data from:

- **Copernicus Marine** — sea temperature, salinity, ocean currents and wave information
- **ERA5** — atmospheric wind and mean sea-level pressure
- **Copernicus bathymetry** — ocean-floor depth

The processed data is intended for **spatial-temporal deep learning models such as GCN/PIGT**, where each ocean location is represented as a graph node and historical observations are used to predict ocean conditions for the next day.

### Study region

```text
Latitude   : 10°N – 20°N
Longitude  : 80°E – 90°E
Grid       : 121 × 121
Total grid cells : 14,641
Final ocean nodes: 12,204
```

---

# 2. Project Data Sources

The raw datasets are stored without modifying the original NetCDF files.

```text
data/
└── raw/
    ├── copernicus_physics/
    │   ├── thetao_bob.nc
    │   ├── so_bob.nc
    │   ├── currents_bob.nc
    │   └── bathymetry.nc
    │
    ├── copernicus_wave/
    │   └── vhm0_bob.nc
    │
    └── era5/
        └── era5_bob.nc
```

### Variables available in the raw data

| Source | Variable | Description |
|---|---|---|
| Copernicus | `thetao` | Sea water temperature |
| Copernicus | `so` | Sea water salinity |
| Copernicus | `uo` | Eastward ocean current |
| Copernicus | `vo` | Northward ocean current |
| Copernicus | `VHM0` | Significant wave height |
| ERA5 | `u10` | 10 m eastward wind |
| ERA5 | `v10` | 10 m northward wind |
| ERA5 | `msl` | Mean sea-level pressure |
| Bathymetry | `deptho` | Sea-floor depth |

The bathymetry file uses the actual Copernicus variable name `deptho`. The ocean mask is derived from valid bathymetry/ocean cells rather than requiring a separate mask variable.

---

# 3. Raw Data Inspection

Before preprocessing, all six NetCDF datasets were inspected for:

- dimensions and variables
- temporal coverage
- timestep spacing
- duplicate timestamps
- latitude/longitude coverage
- depth information
- missing values
- invalid coordinates
- expected variables

The inspection script is:

```text
inspect_raw.py
```

Run it with:

```powershell
python inspect_raw.py
```

It produces:

```text
results/
├── raw_inspection_report.json
└── raw_variable_statistics.csv
```

### Important raw-data findings

Copernicus physics data:

```text
2022-06-01 → 2024-12-31
Daily
945 timesteps
```

Wave data:

```text
2022-11-01 03:00 → 2024-12-31 21:00
3-hourly
```

ERA5:

```text
2022-01-01 → 2024-12-31 18:00
6-hourly
```

All datasets cover the required:

```text
10°N – 20°N
80°E – 90°E
```

There were no duplicate timestamps in the inspected datasets.

Because the wave dataset starts later than the other datasets, the common processing period becomes:

```text
2022-11-01 → 2024-12-31
```

---

# 4. Data Preprocessing

The main preprocessing script is:

```text
data_pipeline.py
```

Run:

```powershell
python data_pipeline.py
```

The pipeline performs the following operations:

```text
Raw NetCDF data
      ↓
Dataset audit
      ↓
Temporal alignment
      ↓
Spatial alignment
      ↓
Ocean-mask construction
      ↓
Invalid/missing-value handling
      ↓
Feature statistics
      ↓
Normalization/scaling
      ↓
7-day sliding windows
      ↓
Chronological split
      ↓
Spatial kNN graph
      ↓
Final dataset packaging
```

### Temporal alignment

The source datasets have different temporal resolutions:

```text
Copernicus physics : daily
Wave               : 3-hourly
ERA5               : 6-hourly
```

The datasets are aligned to a common daily timeline.

The final common period is:

```text
2022-11-01 → 2024-12-31
```

---

# 5. Ocean Mask and Spatial Nodes

The original grid contains:

```text
121 × 121 = 14,641 cells
```

Bathymetry (`deptho`) is used to identify valid ocean locations.

After the spatial validity and preprocessing checks, the final fixed ocean mask contains:

```text
N = 12,204 ocean nodes
```

These same nodes are used consistently for:

- input arrays
- target arrays
- coordinates
- graph construction

This ensures that the node ordering remains consistent throughout the modeling pipeline.

---

# 6. Input Features

The final model input contains **13 features**.

The feature order is fixed as follows:

```text
0   sst
1   salinity
2   u_current
3   v_current
4   swh
5   wind_speed
6   wind_direction
7   pressure
8   bathymetry
9   latitude
10  longitude
11  sin_day_of_year
12  cos_day_of_year
```

### Feature groups

**Ocean variables**

```text
sst
salinity
u_current
v_current
swh
```

**Atmospheric variables**

```text
wind_speed
wind_direction
pressure
```

**Spatial/static information**

```text
bathymetry
latitude
longitude
```

**Temporal information**

```text
sin_day_of_year
cos_day_of_year
```

Wind speed and direction are derived from the ERA5 `u10` and `v10` components.

The final feature order must not be changed when building the model.

---

# 7. Target Variables

The next-day prediction target contains **8 dynamic variables**.

The fixed target order is:

```text
0   sst
1   salinity
2   u_current
3   v_current
4   swh
5   wind_speed
6   wind_direction
7   pressure
```

The target therefore has the shape:

```text
[B, N, 8]
```

where:

- `B` = number of samples
- `N` = 12,204 ocean nodes
- `8` = predicted variables

---

# 8. Sliding-Window Construction

The forecasting setup uses the previous **7 days** to predict the next day.

Conceptually:

```text
t-6  t-5  t-4  t-3  t-2  t-1  t
 |    |    |    |    |    |    |
 └──────────── X ──────────────┘
                         ↓
                       t+1
                         ↓
                         Y
```

Therefore:

```text
X = [B, 7, N, 13]
Y = [B, N, 8]
```

The model receives seven consecutive historical days and predicts the eight dynamic variables for the following day.

---

# 9. Scaling and Normalization

Feature scaling is performed so that variables with larger numerical magnitudes do not dominate model training.

The scaler is fitted using the appropriate training data rather than using future test information.

The fitted scaler is saved in the final data directory and should be reused when processing data for model inference or future experiments.

Do not fit a new scaler separately on the validation or test data.

---

# 10. Chronological Dataset Split

The data is divided chronologically rather than randomly.

### Training

```text
Target dates:
2022-11-08 → 2024-05-09

Samples:
549
```

```text
X_train = (549, 7, 12204, 13)
Y_train = (549, 12204, 8)
```

### Validation

```text
Target dates:
2024-05-10 → 2024-09-03

Samples:
117
```

```text
X_val = (117, 7, 12204, 13)
Y_val = (117, 12204, 8)
```

### Test

```text
Target dates:
2024-09-04 → 2024-12-31

Samples:
119
```

```text
X_test = (119, 7, 12204, 13)
Y_test = (119, 12204, 8)
```

The test period occurs entirely after the training and validation periods, preventing future observations from being mixed into training.

---

# 11. Variability and Data Quality Checks

After preprocessing, the pipeline checks feature variability to identify variables that may have insufficient useful variation.

The pipeline also records the augmentation decision.

Any augmentation experiment must remain **physics-safe** and should not arbitrarily change oceanographic relationships.

No augmentation should be introduced into the model-training workflow unless the saved preprocessing results indicate that it is necessary.

---

# 12. Spatial Graph

A spatial **k-nearest-neighbour (kNN) graph** is created using the geographic coordinates of the 12,204 ocean nodes.

The graph represents spatial relationships between nearby ocean locations and is intended for the GCN/PIGT component.

The graph is saved as:

```text
data/
└── graph/
    └── graph.pt
```

The corresponding node coordinates are saved with the final dataset.

The graph must use the same node ordering as the X/Y tensors.

---

# 13. Final Data Directory

After running the complete pipeline, the processed project contains a final data package under:

```text
data/
├── raw/
│   └── ...
│
├── final/
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── Y_train.npy
│   ├── Y_val.npy
│   ├── Y_test.npy
│   ├── scaler files
│   ├── metadata.json
│   ├── feature/target information
│   ├── coordinates
│   ├── split information
│   ├── augmentation decision
│   └── README
│
└── graph/
    └── graph.pt
```

The exact filenames should be taken from the generated `metadata.json`/final directory rather than assumed by a downstream script.

---

# 14. Current Verified Dataset

The preprocessing pipeline has completed successfully with:

```text
FINAL PACKAGE VERIFICATION: PASS
```

Current dataset:

```text
Common period : 2022-11-01 → 2024-12-31
Ocean nodes   : 12,204
Input features: 13
Target variables: 8
History       : 7 days
Prediction    : next day
```

Tensor structure:

```text
X_train = (549, 7, 12204, 13)
X_val   = (117, 7, 12204, 13)
X_test  = (119, 7, 12204, 13)

Y_train = (549, 12204, 8)
Y_val   = (117, 12204, 8)
Y_test  = (119, 12204, 8)
```

The data-preparation stage is therefore ready for the next stage of the project: **model development and training**.

---

# 15. Project Scripts

Current important scripts:

```text
inspect_raw.py
```

Checks the original NetCDF files.

```text
pigt_config.py
```

Contains project paths and preprocessing configuration.

```text
data_pipeline.py
```

Runs the complete data preparation pipeline and creates the final dataset and graph.

---

# 16. How to Continue From Here

The data preparation is already complete, so **do not download or preprocess the raw datasets again**.

First activate the environment:

```powershell
venv\Scripts\activate
```

The next stage can start by loading:

```text
data/final/
data/graph/graph.pt
```

and checking `metadata.json` to obtain the exact saved filenames, feature order, target order, node information and split details.

Then verify that:

```text
X → [B, 7, 12204, 13]
Y → [B, 12204, 8]
Graph → 12,204 nodes
```

After that, the next stage can focus on the **GCN/PIGT model architecture, training, validation, loss functions, and forecasting evaluation**.

In simple terms: **the raw-data work is finished; start from the saved `data/final` arrays and `data/graph/graph.pt`, and build the model on top of them.**