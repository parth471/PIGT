# PIGT Final Dataset

## Purpose
Training-ready daily ocean-environment dataset for the PIGT models.

## Shapes
- `X_train`, `X_val`, `X_test`: `[B, 7, N, 13]`
- `Y_train`, `Y_val`, `Y_test`: `[B, N, 8]`
- `N = 12204` ocean nodes

## Input feature order
```text
0: sst
1: salinity
2: u_current
3: v_current
4: swh
5: wind_speed
6: wind_direction
7: pressure
8: bathymetry
9: latitude
10: longitude
11: sin_day_of_year
12: cos_day_of_year
```

## Target order
```text
0: sst
1: salinity
2: u_current
3: v_current
4: swh
5: wind_speed
6: wind_direction
7: pressure
```

## Time protocol
Seven historical daily steps `t-6 ... t` are used to predict the next day `t+1`.
The splits are chronological 70/15/15 by target date. Future validation/test periods are not used to fit the scalers.

## Raw data alignment
- Copernicus SST, salinity and surface currents are already daily on the 121x121 ocean grid.
- Copernicus Wave `VHM0` is aggregated from 3-hourly to daily and interpolated to the ocean grid.
- ERA5 `u10`, `v10`, and `msl` are aggregated from 6-hourly to daily and interpolated to the ocean grid.
- Wind speed and meteorological wind direction are derived from `u10`/`v10`.
- ERA5 pressure is stored in hPa.
- Bathymetry uses the supplied `deptho` variable; finite positive `deptho` defines the ocean mask.

## Cleaning
Only temporal gaps of up to 3 consecutive days are linearly interpolated. Land cells are never filled numerically. Nodes that remain incomplete in any target variable after cleaning are excluded so every final sample has a fixed N.

## Files
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `Y_train.npy`, `Y_val.npy`, `Y_test.npy`
- `feature_scaler.pkl` — feature and target scaling metadata
- `metadata.json` — complete dataset contract, feature order, target order, split dates and graph metadata
- `coordinates.npy`, `node_lat.npy`, `node_lon.npy` — graph node coordinates in exactly the same N order as X/Y
- `node_mask.npy` — full 121x121 final ocean-node mask
- Graph: `data/graph/graph.pt`

## Load example
```python
import pickle
from pathlib import Path
import numpy as np
import torch

final = Path('data/final')
X_train = np.load(final / 'X_train.npy')
Y_train = np.load(final / 'Y_train.npy')

with open(final / 'feature_scaler.pkl', 'rb') as f:
    scalers = pickle.load(f)

graph = torch.load('data/graph/graph.pt', map_location='cpu')

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('edges:', graph['edge_index'].shape)
```

## Augmentation
Decision: `NO_AUGMENTATION`. No augmentation is applied automatically. Any augmentation must be a separate controlled experiment and compared against the untouched baseline.
