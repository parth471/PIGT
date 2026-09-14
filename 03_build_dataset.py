"""
STEP 10, 11, 12, 15
Build the final ML dataset from the merged Bay of Bengal data.

Input:
    data/processed/merged_bob_daily.nc

Final common period:
    2022-11-01 to 2024-12-31

Final common ocean nodes:
    12,204

Output:
    data/final/
        X_train.npy
        Y_train.npy
        X_val.npy
        Y_val.npy
        X_test.npy
        Y_test.npy
        lat.npy
        lon.npy
        node_mask.npy
        node_lat.npy
        node_lon.npy
        feature_scaler.pkl
        metadata.json

USAGE:
    python 03_build_dataset.py
"""

import json
import pickle

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from config import (
    MERGED_NC_PATH,
    FINAL_DIR,
    HISTORY_DAYS,
    FORECAST_HORIZON_DAYS,
    TRAIN_FRAC,
    VAL_FRAC,
    FEATURE_ORDER,
    TARGET_ORDER,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)


# ============================================================
# COMMON DATA PERIOD
# ============================================================

COMMON_START_DATE = "2022-11-01"
COMMON_END_DATE = "2024-12-31"


# ============================================================
# BUILD COMMON VALID OCEAN NODE MASK
# ============================================================

def build_common_node_mask(ds):
    """
    Build the final spatial mask.

    A node is included only if:
        1. It is ocean according to bathymetry.
        2. SST is valid for the complete common period.
        3. Salinity is valid for the complete common period.
        4. U current is valid for the complete common period.
        5. V current is valid for the complete common period.
        6. SWH is valid for the complete common period.

    Returns:
        lat_idx
        lon_idx
        mask_bool
    """

    print("\nBuilding common valid ocean-node mask...")

    ocean_mask = ds["land_mask"] == 1

    required_variables = [
        "sst",
        "salinity",
        "u_current",
        "v_current",
        "swh",
    ]

    common_mask = ocean_mask.copy()

    for var in required_variables:

        print(f"  Checking {var}...")

        complete = ds[var].notnull().all(dim="time")

        common_mask = common_mask & complete

    mask_bool = common_mask.values.astype(bool)

    lat_idx, lon_idx = np.where(mask_bool)

    print(f"\nFinal common ocean nodes: {len(lat_idx)}")

    if len(lat_idx) != 12204:
        print(
            "WARNING: Expected approximately 12,204 nodes "
            f"based on the previous coverage check, but found {len(lat_idx)}."
        )

    return lat_idx, lon_idx, mask_bool


# ============================================================
# EXTRACT NODE SERIES
# ============================================================

def extract_node_series(data_array, lat_idx, lon_idx):
    """
    Convert:

        [time, latitude, longitude]

    into:

        [time, nodes]
    """

    values = data_array.values

    return values[:, lat_idx, lon_idx]


# ============================================================
# DAY-OF-YEAR FEATURES
# ============================================================

def day_of_year_features(time_values):

    time_da = xr.DataArray(time_values)

    doy = (
        time_da.dt.dayofyear
        .values
        .astype(float)
    )

    sin_doy = np.sin(
        2 * np.pi * doy / 365.25
    )

    cos_doy = np.cos(
        2 * np.pi * doy / 365.25
    )

    return sin_doy, cos_doy


# ============================================================
# BUILD TEMPORAL SEQUENCES
# ============================================================

def build_sequences(
    feature_array,
    target_array,
    history,
    horizon,
):
    """
    feature_array:
        [T, N, F]

    target_array:
        [T, N, 5]

    X:
        [B, history, N, F]

    Y:
        [B, N, 5]
    """

    T = feature_array.shape[0]

    B = (
        T
        - history
        - horizon
        + 1
    )

    if B <= 0:

        raise ValueError(
            f"Not enough timesteps ({T}) for "
            f"history={history} and "
            f"horizon={horizon}."
        )

    X = np.stack(
        [
            feature_array[
                i:i + history
            ]
            for i in range(B)
        ]
    )

    Y = np.stack(
        [
            target_array[
                i + history + horizon - 1
            ]
            for i in range(B)
        ]
    )

    return X, Y


# ============================================================
# FIT SCALER
# ============================================================

def fit_scaler_on_slice(
    channel_array,
    cutoff_time_idx,
):
    """
    Fit StandardScaler using only
    the training-period data.
    """

    train_slice = channel_array[
        :cutoff_time_idx
    ]

    flat = train_slice.reshape(-1, 1)

    flat = flat[
        ~np.isnan(flat)
    ].reshape(-1, 1)

    if len(flat) == 0:

        raise ValueError(
            "No valid values available "
            "for fitting the scaler."
        )

    scaler = StandardScaler()

    scaler.fit(flat)

    return scaler


# ============================================================
# APPLY SCALER
# ============================================================

def apply_scaler(
    channel_array,
    scaler,
):

    original_shape = channel_array.shape

    flat = channel_array.reshape(
        -1, 1
    ).astype(float)

    valid = ~np.isnan(
        flat
    ).flatten()

    out = flat.copy()

    if valid.any():

        out[valid, 0] = (
            scaler.transform(
                flat[valid].reshape(-1, 1)
            )
            .flatten()
        )

    return out.reshape(
        original_shape
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PIGT — BUILD FINAL ML DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading merged dataset...")

    ds = xr.open_dataset(
        MERGED_NC_PATH
    )

    print(
        "Original dataset dimensions:",
        dict(ds.sizes)
    )

    # --------------------------------------------------------
    # SELECT COMMON PERIOD
    # --------------------------------------------------------

    print(
        f"\nSelecting common period: "
        f"{COMMON_START_DATE} → {COMMON_END_DATE}"
    )

    ds = ds.sel(
        time=slice(
            COMMON_START_DATE,
            COMMON_END_DATE,
        )
    )

    print(
        "Selected dataset dimensions:",
        dict(ds.sizes)
    )

    time_values = ds.time.values

    T = len(time_values)

    print(
        f"Number of daily timesteps: {T}"
    )

    # --------------------------------------------------------
    # BUILD COMMON NODE MASK
    # --------------------------------------------------------

    (
        lat_idx,
        lon_idx,
        mask_bool,
    ) = build_common_node_mask(ds)

    n_nodes = len(lat_idx)

    # --------------------------------------------------------
    # GRID COORDINATES
    # --------------------------------------------------------

    lat_vals_full = (
        ds.latitude.values
    )

    lon_vals_full = (
        ds.longitude.values
    )

    node_lat = (
        lat_vals_full[lat_idx]
    )

    node_lon = (
        lon_vals_full[lon_idx]
    )

    print(
        "Grid shape:",
        mask_bool.shape
    )

    print(
        "Number of final nodes:",
        n_nodes
    )

    # --------------------------------------------------------
    # EXTRACT VARIABLES
    # --------------------------------------------------------

    print(
        "\nExtracting per-node time series..."
    )

    series = {}

    variables = [
        "sst",
        "salinity",
        "u_current",
        "v_current",
        "swh",
        "wind_speed",
        "wind_direction",
        "pressure",
    ]

    for var in variables:

        print(f"  {var}")

        series[var] = extract_node_series(
            ds[var],
            lat_idx,
            lon_idx,
        )

    # --------------------------------------------------------
    # STATIC NODE FEATURES
    # --------------------------------------------------------

    bathymetry_node = (
        ds["bathymetry"]
        .values[
            lat_idx,
            lon_idx
        ]
    )

    bathymetry_full = np.repeat(
        bathymetry_node[
            np.newaxis,
            :
        ],
        T,
        axis=0,
    )

    lat_full = np.repeat(
        node_lat[
            np.newaxis,
            :
        ],
        T,
        axis=0,
    )

    lon_full = np.repeat(
        node_lon[
            np.newaxis,
            :
        ],
        T,
        axis=0,
    )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    sin_doy, cos_doy = (
        day_of_year_features(
            time_values
        )
    )

    sin_doy_full = np.repeat(
        sin_doy[:, np.newaxis],
        n_nodes,
        axis=1,
    )

    cos_doy_full = np.repeat(
        cos_doy[:, np.newaxis],
        n_nodes,
        axis=1,
    )

    # --------------------------------------------------------
    # BUILD FEATURE ARRAY
    # --------------------------------------------------------

    feature_components = {

        "sst":
            series["sst"],

        "salinity":
            series["salinity"],

        "u_current":
            series["u_current"],

        "v_current":
            series["v_current"],

        "swh":
            series["swh"],

        "wind_speed":
            series["wind_speed"],

        "wind_direction":
            series["wind_direction"],

        "pressure":
            series["pressure"],

        "bathymetry":
            bathymetry_full,

        "latitude":
            lat_full,

        "longitude":
            lon_full,

        "sin_day_of_year":
            sin_doy_full,

        "cos_day_of_year":
            cos_doy_full,
    }

    X_full = np.stack(
        [
            feature_components[name]
            for name in FEATURE_ORDER
        ],
        axis=-1,
    )

    # --------------------------------------------------------
    # BUILD TARGET ARRAY
    # --------------------------------------------------------

    target_components = {

        "sst":
            series["sst"],

        "salinity":
            series["salinity"],

        "u_current":
            series["u_current"],

        "v_current":
            series["v_current"],

        "swh":
            series["swh"],
    }

    Y_full = np.stack(
        [
            target_components[name]
            for name in TARGET_ORDER
        ],
        axis=-1,
    )

    print(
        "\nX_full shape:",
        X_full.shape
    )

    print(
        "Y_full shape:",
        Y_full.shape
    )

    # --------------------------------------------------------
    # FINAL NaN CHECK
    # --------------------------------------------------------

    x_nan_count = int(
        np.isnan(X_full).sum()
    )

    y_nan_count = int(
        np.isnan(Y_full).sum()
    )

    print(
        "\nNaNs in X before normalization:",
        x_nan_count
    )

    print(
        "NaNs in Y before normalization:",
        y_nan_count
    )

    if y_nan_count > 0:

        raise ValueError(
            "Target data still contains NaNs. "
            "Do not continue until this is investigated."
        )

    # --------------------------------------------------------
    # TRAINING PERIOD CUTOFF
    # --------------------------------------------------------

    train_cutoff_time = int(
        T * TRAIN_FRAC
    )

    print(
        "\nTraining-period cutoff:",
        train_cutoff_time,
        "of",
        T,
        "days"
    )

    # --------------------------------------------------------
    # NORMALIZE FEATURES
    # --------------------------------------------------------

    print(
        "\nNormalizing feature channels..."
    )

    feature_scalers = {}

    minmax_features = {
        "latitude",
        "longitude",
    }

    for i, name in enumerate(
        FEATURE_ORDER
    ):

        channel = X_full[:, :, i]

        if name in minmax_features:

            cmin = float(
                np.nanmin(channel)
            )

            cmax = float(
                np.nanmax(channel)
            )

            if cmax == cmin:

                X_full[:, :, i] = 0.0

            else:

                X_full[:, :, i] = (
                    (
                        channel - cmin
                    )
                    /
                    (cmax - cmin)
                    * 2
                    - 1
                )

            feature_scalers[name] = {
                "type": "minmax",
                "min": cmin,
                "max": cmax,
            }

        else:

            scaler = (
                fit_scaler_on_slice(
                    channel,
                    train_cutoff_time,
                )
            )

            X_full[:, :, i] = (
                apply_scaler(
                    channel,
                    scaler,
                )
            )

            feature_scalers[name] = scaler

    # --------------------------------------------------------
    # NORMALIZE TARGETS
    # --------------------------------------------------------

    print(
        "Normalizing target channels..."
    )

    target_scalers = {}

    for i, name in enumerate(
        TARGET_ORDER
    ):

        channel = Y_full[:, :, i]

        scaler = (
            fit_scaler_on_slice(
                channel,
                train_cutoff_time,
            )
        )

        Y_full[:, :, i] = (
            apply_scaler(
                channel,
                scaler,
            )
        )

        target_scalers[name] = scaler

    # --------------------------------------------------------
    # BUILD TEMPORAL SEQUENCES
    # --------------------------------------------------------

    print(
        f"\nBuilding sequences:"
        f" history={HISTORY_DAYS} days,"
        f" horizon={FORECAST_HORIZON_DAYS} day(s)"
    )

    X_seq, Y_seq = build_sequences(
        X_full,
        Y_full,
        HISTORY_DAYS,
        FORECAST_HORIZON_DAYS,
    )

    print(
        "X_seq shape:",
        X_seq.shape
    )

    print(
        "Y_seq shape:",
        Y_seq.shape
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    B = X_seq.shape[0]

    train_end = int(
        B * TRAIN_FRAC
    )

    val_end = int(
        B *
        (
            TRAIN_FRAC
            +
            VAL_FRAC
        )
    )

    X_train = X_seq[
        :train_end
    ]

    Y_train = Y_seq[
        :train_end
    ]

    X_val = X_seq[
        train_end:val_end
    ]

    Y_val = Y_seq[
        train_end:val_end
    ]

    X_test = X_seq[
        val_end:
    ]

    Y_test = Y_seq[
        val_end:
    ]

    print(
        "\nDataset split:"
    )

    print(
        "Train:",
        X_train.shape,
        Y_train.shape
    )

    print(
        "Validation:",
        X_val.shape,
        Y_val.shape
    )

    print(
        "Test:",
        X_test.shape,
        Y_test.shape
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SAVE ARRAYS
    # --------------------------------------------------------

    print(
        "\nSaving final arrays to:",
        FINAL_DIR
    )

    np.save(
        FINAL_DIR / "X_train.npy",
        X_train.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "Y_train.npy",
        Y_train.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "X_val.npy",
        X_val.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "Y_val.npy",
        Y_val.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "X_test.npy",
        X_test.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "Y_test.npy",
        Y_test.astype(np.float32),
    )

    np.save(
        FINAL_DIR / "lat.npy",
        lat_vals_full,
    )

    np.save(
        FINAL_DIR / "lon.npy",
        lon_vals_full,
    )

    np.save(
        FINAL_DIR / "node_mask.npy",
        mask_bool,
    )

    np.save(
        FINAL_DIR / "node_lat.npy",
        node_lat,
    )

    np.save(
        FINAL_DIR / "node_lon.npy",
        node_lon,
    )

    # --------------------------------------------------------
    # SAVE SCALERS
    # --------------------------------------------------------

    with open(
        FINAL_DIR / "feature_scaler.pkl",
        "wb",
    ) as f:

        pickle.dump(
            {
                "features":
                    feature_scalers,

                "targets":
                    target_scalers,
            },
            f,
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "region": {
            "lat_min":
                LAT_MIN,

            "lat_max":
                LAT_MAX,

            "lon_min":
                LON_MIN,

            "lon_max":
                LON_MAX,
        },

        "time_range": {
            "start":
                COMMON_START_DATE,

            "end":
                COMMON_END_DATE,
        },

        "temporal_resolution":
            "daily",

        "total_timesteps":
            int(T),

        "history_window_days":
            HISTORY_DAYS,

        "forecast_horizon_days":
            FORECAST_HORIZON_DAYS,

        "feature_order":
            FEATURE_ORDER,

        "target_order":
            TARGET_ORDER,

        "n_ocean_nodes":
            int(n_nodes),

        "grid_shape":
            list(mask_bool.shape),

        "split_fractions": {

            "train":
                TRAIN_FRAC,

            "val":
                VAL_FRAC,

            "test":
                round(
                    1
                    - TRAIN_FRAC
                    - VAL_FRAC,
                    4,
                ),
        },

        "n_sequences": {

            "train":
                int(
                    X_train.shape[0]
                ),

            "val":
                int(
                    X_val.shape[0]
                ),

            "test":
                int(
                    X_test.shape[0]
                ),
        },

        "normalization":
            (
                "StandardScaler per "
                "feature/target channel, "
                "fit using training period "
                "only. Latitude and longitude "
                "use min-max scaling to [-1, 1]."
            ),

        "common_node_definition":
            (
                "Ocean nodes with complete "
                "SST, salinity, U current, "
                "V current, and SWH coverage "
                "throughout the common period."
            ),
    }

    with open(
        FINAL_DIR / "metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print(
        "\nChecking final arrays for NaNs..."
    )

    print(
        "X_train NaNs:",
        int(np.isnan(X_train).sum())
    )

    print(
        "Y_train NaNs:",
        int(np.isnan(Y_train).sum())
    )

    print(
        "X_val NaNs:",
        int(np.isnan(X_val).sum())
    )

    print(
        "Y_val NaNs:",
        int(np.isnan(Y_val).sum())
    )

    print(
        "X_test NaNs:",
        int(np.isnan(X_test).sum())
    )

    print(
        "Y_test NaNs:",
        int(np.isnan(Y_test).sum())
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET BUILD COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nFiles written to:",
        FINAL_DIR
    )

    for file_path in sorted(
        FINAL_DIR.glob("*")
    ):

        print(
            " -",
            file_path.name
        )


if __name__ == "__main__":
    main()