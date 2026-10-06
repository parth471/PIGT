"""
PIGT project data-preparation pipeline.

This script builds the complete training-ready data package from the raw
Copernicus Marine, Copernicus Wave, ERA5 and bathymetry NetCDF files already
present in data/raw/.

Run from the project root:
    python data_pipeline.py

Outputs:
    data/processed/merged_bob_daily_clean.nc
    data/final/X_train.npy
    data/final/X_val.npy
    data/final/X_test.npy
    data/final/Y_train.npy
    data/final/Y_val.npy
    data/final/Y_test.npy
    data/final/feature_scaler.pkl
    data/final/metadata.json
    data/final/coordinates.npy
    data/final/node_mask.npy
    data/graph/graph.pt
    results/data_pipeline/*

Important:
    - Bathymetry uses the actual variable name `deptho`.
    - ERA5 is 6-hourly and is aggregated to daily values.
    - Wave data is 3-hourly and is aggregated to daily values.
    - The common period is calculated from the raw files, then clipped to the
      project bounds. With the supplied files this should begin at 2022-11-01.
    - No random temporal split is used.
    - Scalers are fitted using training-period data only.
    - Persistent missing ocean nodes are excluded; land is never filled with 0.
"""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pigt_config import *


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def find_lat_name(ds: xr.Dataset) -> str:
    for name in ("latitude", "lat"):
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find latitude coordinate. Found: {list(ds.coords)}")


def find_lon_name(ds: xr.Dataset) -> str:
    for name in ("longitude", "lon"):
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find longitude coordinate. Found: {list(ds.coords)}")


def find_time_name(ds: xr.Dataset) -> str | None:
    for name in ("time", "valid_time"):
        if name in ds.coords:
            return name
    return None


def normalise_time(ds: xr.Dataset) -> xr.Dataset:
    time_name = find_time_name(ds)
    if time_name is None:
        return ds
    if time_name != "time":
        ds = ds.rename({time_name: "time"})
    return ds.sortby("time")


def deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    t = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    keep = ~t.duplicated(keep="first")
    if not keep.all():
        ds = ds.isel(time=np.flatnonzero(keep))
    return ds


def ensure_sorted_spatial(ds: xr.Dataset) -> xr.Dataset:
    """Sort latitude/longitude so xarray interpolation is reliable."""
    lat = find_lat_name(ds)
    lon = find_lon_name(ds)
    if ds[lat].ndim == 1 and ds[lat].size > 1:
        ds = ds.sortby(lat)
    if ds[lon].ndim == 1 and ds[lon].size > 1:
        ds = ds.sortby(lon)
    return ds


def surface(da: xr.DataArray) -> xr.DataArray:
    """Select the first depth level if a depth dimension is present."""
    for depth_name in ("depth", "deptht", "lev", "level"):
        if depth_name in da.dims:
            if da.sizes[depth_name] != 1:
                print(f"WARNING: {da.name} has {da.sizes[depth_name]} depth levels; using first level.")
            da = da.isel({depth_name: 0})
            break
    return da


def to_float32(da: xr.DataArray) -> xr.DataArray:
    return da.astype(np.float32)


# -----------------------------------------------------------------------------
# Raw audit
# -----------------------------------------------------------------------------

def audit_dataset(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    with xr.open_dataset(path) as ds:
        ds = ensure_sorted_spatial(ds)
        lat_name = find_lat_name(ds)
        lon_name = find_lon_name(ds)
        time_name = find_time_name(ds)

        result = {
            "label": label,
            "file": str(path),
            "dimensions": {k: int(v) for k, v in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "variables": list(ds.data_vars),
            "latitude": {
                "name": lat_name,
                "min": float(np.nanmin(ds[lat_name].values)),
                "max": float(np.nanmax(ds[lat_name].values)),
                "count": int(ds[lat_name].size),
            },
            "longitude": {
                "name": lon_name,
                "min": float(np.nanmin(ds[lon_name].values)),
                "max": float(np.nanmax(ds[lon_name].values)),
                "count": int(ds[lon_name].size),
            },
            "time": None,
            "variable_statistics": {},
        }

        if time_name is not None:
            times = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
            diffs = pd.Series(times).diff().dropna().dt.total_seconds() / 3600.0
            result["time"] = {
                "coordinate": time_name,
                "count": int(len(times)),
                "start": str(times.min()),
                "end": str(times.max()),
                "duplicate_timestamps": int(times.duplicated().sum()),
                "monotonic_increasing": bool(times.is_monotonic_increasing),
                "unique_step_hours": sorted({float(x) for x in diffs.unique()}),
            }

        for name, da in ds.data_vars.items():
            values = np.asarray(da.values)
            numeric = np.issubdtype(values.dtype, np.number)
            if numeric:
                finite = np.isfinite(values)
                missing = int(np.isnan(values).sum())
                inf_count = int(np.isinf(values).sum())
                finite_values = values[finite]
                stats = {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "units": da.attrs.get("units"),
                    "long_name": da.attrs.get("long_name"),
                    "standard_name": da.attrs.get("standard_name"),
                    "missing_count": missing,
                    "missing_percent": float(100.0 * missing / values.size) if values.size else 0.0,
                    "infinite_count": inf_count,
                }
                if finite_values.size:
                    stats.update({
                        "min": float(np.min(finite_values)),
                        "max": float(np.max(finite_values)),
                        "mean": float(np.mean(finite_values)),
                        "std": float(np.std(finite_values)),
                    })
                result["variable_statistics"][name] = stats

        # Bathymetry-specific check.
        if label == "bathymetry":
            result["bathymetry_variable"] = "deptho" if "deptho" in ds.data_vars else None

    return result


def audit_all_raw() -> dict:
    print("\n[1/9] Auditing raw NetCDF files...")
    report = {"files": {}}
    for label, path in RAW_FILES.items():
        print(f"  - {label}: {path.name}")
        report["files"][label] = audit_dataset(path, label)

    # Check the common spatial coverage.
    report["expected_region"] = {
        "latitude": [LAT_MIN, LAT_MAX],
        "longitude": [LON_MIN, LON_MAX],
    }

    save_json(REPORT_DIR / "raw_audit.json", report)
    return report


# -----------------------------------------------------------------------------
# Loading and common-time calculation
# -----------------------------------------------------------------------------

def get_dataset_time_bounds(path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    with xr.open_dataset(path) as ds:
        ds = normalise_time(ds)
        if "time" not in ds.coords:
            raise ValueError(f"No time coordinate in {path}")
        t = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
        return t.min(), t.max()


def determine_common_period() -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for label, path in RAW_FILES.items():
        if label == "bathymetry":
            continue
        start, end = get_dataset_time_bounds(path)
        starts.append(start)
        ends.append(end)

    common_start = max(starts)
    common_end = min(ends)

    # Clip to project-level bounds.
    project_start = pd.Timestamp(PROJECT_START)
    project_end = pd.Timestamp(PROJECT_END) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    common_start = max(common_start, project_start)
    common_end = min(common_end, project_end)

    if common_start >= common_end:
        raise RuntimeError(f"No common temporal period. Calculated {common_start} to {common_end}.")

    return common_start.normalize(), common_end.normalize()


def open_dynamic_datasets():
    print("\n[2/9] Loading and aligning source datasets...")
    thetao = ensure_sorted_spatial(deduplicate_time(normalise_time(xr.open_dataset(RAW_FILES["sst"]))))
    sal = ensure_sorted_spatial(deduplicate_time(normalise_time(xr.open_dataset(RAW_FILES["salinity"]))))
    cur = ensure_sorted_spatial(deduplicate_time(normalise_time(xr.open_dataset(RAW_FILES["currents"]))))
    wave = ensure_sorted_spatial(deduplicate_time(normalise_time(xr.open_dataset(RAW_FILES["wave"]))))
    era = ensure_sorted_spatial(deduplicate_time(normalise_time(xr.open_dataset(RAW_FILES["era5"]))))
    bath = ensure_sorted_spatial(xr.open_dataset(RAW_FILES["bathymetry"]))

    return thetao, sal, cur, wave, era, bath


# -----------------------------------------------------------------------------
# Alignment / cleaning
# -----------------------------------------------------------------------------

def load_and_align() -> Tuple[xr.Dataset, pd.Timestamp, pd.Timestamp]:
    thetao, sal, cur, wave, era, bath = open_dynamic_datasets()

    try:
        common_start, common_end = determine_common_period()
        print(f"  Common period: {common_start.date()} -> {common_end.date()}")

        ocean_lat_name = find_lat_name(thetao)
        ocean_lon_name = find_lon_name(thetao)
        ocean_lat = thetao[ocean_lat_name]
        ocean_lon = thetao[ocean_lon_name]

        # --- Copernicus Physics ---
        sst = surface(thetao["thetao"]).rename("sst")
        salinity = surface(sal["so"]).rename("salinity")
        u_current = surface(cur["uo"]).rename("u_current")
        v_current = surface(cur["vo"]).rename("v_current")

        # --- Actual bathymetry variable in the supplied file ---
        if "deptho" not in bath.data_vars:
            raise ValueError(
                "bathymetry.nc does not contain `deptho`. "
                f"Available variables: {list(bath.data_vars)}"
            )
        bath_lat_name = find_lat_name(bath)
        bath_lon_name = find_lon_name(bath)
        bathymetry = bath["deptho"].rename("bathymetry")
        bathymetry = bathymetry.interp(
            {bath_lat_name: ocean_lat, bath_lon_name: ocean_lon},
            method="nearest",
        )
        bathymetry = bathymetry.rename({bath_lat_name: "latitude", bath_lon_name: "longitude"})
        # A valid positive seafloor depth is treated as ocean. NaN/zero is land/non-ocean.
        ocean_mask = (np.isfinite(bathymetry) & (bathymetry > 0)).rename("ocean_mask")

        # --- Copernicus Wave: 3-hourly -> daily mean -> ocean grid ---
        swh = wave["VHM0"].resample(time=FREQUENCY).mean()
        swh = swh.interp(latitude=ocean_lat, longitude=ocean_lon, method="linear")
        swh = swh.rename("swh")

        # --- ERA5: 6-hourly -> daily mean -> ocean grid ---
        u10 = era["u10"].resample(time=FREQUENCY).mean()
        v10 = era["v10"].resample(time=FREQUENCY).mean()
        msl = era["msl"].resample(time=FREQUENCY).mean()

        u10 = u10.interp(latitude=ocean_lat, longitude=ocean_lon, method="linear")
        v10 = v10.interp(latitude=ocean_lat, longitude=ocean_lon, method="linear")
        msl = msl.interp(latitude=ocean_lat, longitude=ocean_lon, method="linear")

        # Wind speed in m/s.
        wind_speed = np.hypot(u10, v10).rename("wind_speed")

        # Meteorological wind direction (direction from which the wind blows), degrees clockwise from north.
        wind_direction = ((180.0 + np.degrees(np.arctan2(u10, v10))) % 360.0).rename("wind_direction")

        # ERA5 MSL is normally Pa; convert to hPa only when necessary.
        msl_units = str(era["msl"].attrs.get("units", "")).lower()
        if "pa" in msl_units and "hpa" not in msl_units:
            pressure = (msl / 100.0).rename("pressure")
            pressure_units = "hPa"
        else:
            # Fallback based on magnitude if metadata is missing.
            median_pressure = float(msl.median(skipna=True).values)
            if median_pressure > 2000:
                pressure = (msl / 100.0).rename("pressure")
                pressure_units = "hPa"
            else:
                pressure = msl.rename("pressure")
                pressure_units = msl_units or "unknown"

        # --- Common daily intersection ---
        ds = xr.Dataset({
            "sst": sst,
            "salinity": salinity,
            "u_current": u_current,
            "v_current": v_current,
            "swh": swh,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "pressure": pressure,
            "bathymetry": bathymetry,
            "ocean_mask": ocean_mask,
        })

        # Restrict to the common period and exact daily dates.
        ds = ds.sel(time=slice(common_start, common_end))
        ds = ds.sortby("time")

        # Ensure all variables are on exactly the same 1-degree? No: preserve the
        # supplied 0.0833-degree 121x121 ocean grid.
        ds["ocean_mask"] = ds["ocean_mask"].astype(bool)

        # Existing cleaning rule: linearly interpolate only short temporal gaps.
        # Land is masked first so land NaNs are never filled.
        for variable in TARGET_ORDER:
            masked = ds[variable].where(ds["ocean_mask"])
            ds[variable] = masked.interpolate_na(
                dim="time",
                method="linear",
                limit=MAX_GAP_DAYS,
            )

        ds.attrs["pressure_units"] = pressure_units
        ds.attrs["common_period_start"] = str(common_start.date())
        ds.attrs["common_period_end"] = str(common_end.date())
        ds.attrs["cleaning_rule"] = f"Linear interpolation for temporal gaps <= {MAX_GAP_DAYS} days; persistent missing ocean nodes excluded."

        ds = ds.load()
        ds.to_netcdf(PROCESSED_DIR / "merged_bob_daily_clean.nc")
        return ds, common_start, common_end

    finally:
        for obj in (thetao, sal, cur, wave, era, bath):
            try:
                obj.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Invalid-cell and feature statistics
# -----------------------------------------------------------------------------

def invalid_cell_audit(ds: xr.Dataset) -> dict:
    print("\n[3/9] Checking invalid values over the ocean mask...")
    ocean = ds["ocean_mask"].values.astype(bool)
    report = {}

    for variable, bounds in VALID_RANGES.items():
        if variable == "bathymetry":
            values = ds["bathymetry"].values
            mask = ocean
        else:
            values = ds[variable].values
            mask = np.broadcast_to(ocean, values.shape)

        ocean_values = values[mask]
        finite = np.isfinite(ocean_values)
        finite_values = ocean_values[finite]
        lo, hi = bounds
        out_of_range = finite_values[(finite_values < lo) | (finite_values > hi)]

        report[variable] = {
            "ocean_cells_checked": int(ocean_values.size),
            "nonfinite_ocean_count": int((~finite).sum()),
            "nonfinite_ocean_percent": float(100.0 * (~finite).mean()) if ocean_values.size else 0.0,
            "out_of_range_count": int(out_of_range.size),
            "out_of_range_percent": float(100.0 * out_of_range.size / finite_values.size) if finite_values.size else 0.0,
            "lower_bound": lo,
            "upper_bound": hi,
        }

    save_json(REPORT_DIR / "invalid_cells.json", report)
    return report


def build_common_ocean_nodes(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a fixed node set.

    1. Start from bathymetry ocean mask.
    2. Require all eight dynamic target variables to be finite for every day after
       short-gap cleaning.
    3. Keep the same N nodes for every sample.
    """
    print("\n[4/9] Building the final fixed ocean-node mask...")

    mask = ds["ocean_mask"].values.astype(bool)

    for variable in TARGET_ORDER:
        finite_all_days = np.isfinite(ds[variable].values).all(axis=0)
        mask &= finite_all_days

    lat_idx, lon_idx = np.where(mask)
    if len(lat_idx) == 0:
        raise RuntimeError("No valid ocean nodes remain after the common completeness check.")

    print(f"  Full grid: {mask.shape[0]} x {mask.shape[1]} = {mask.size} cells")
    print(f"  Final ocean nodes N = {len(lat_idx)}")

    return lat_idx, lon_idx, mask


def extract_time_space(da: xr.DataArray, lat_idx, lon_idx) -> np.ndarray:
    return np.asarray(da.values[:, lat_idx, lon_idx], dtype=np.float64)


def temporal_features(times: pd.DatetimeIndex, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    doy = times.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * doy / 365.25
    sin = np.repeat(np.sin(angle)[:, None], n_nodes, axis=1)
    cos = np.repeat(np.cos(angle)[:, None], n_nodes, axis=1)
    return sin, cos


def calculate_feature_statistics(ds: xr.Dataset, lat_idx, lon_idx, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Calculate min/max/mean/std/missing/outliers/temporal variation for all 13 inputs."""
    print("\n[5/9] Calculating statistics for all 13 input features...")

    rows = []
    dynamic_and_static = [
        "sst", "salinity", "u_current", "v_current", "swh",
        "wind_speed", "wind_direction", "pressure", "bathymetry",
    ]

    for name in dynamic_and_static:
        if name == "bathymetry":
            arr = np.asarray(ds[name].values[lat_idx, lon_idx], dtype=np.float64)[None, :]
            arr = np.repeat(arr, len(times), axis=0)
        else:
            arr = extract_time_space(ds[name], lat_idx, lon_idx)

        flat = arr.reshape(-1)
        finite = np.isfinite(flat)
        values = flat[finite]

        if values.size == 0:
            q1 = q3 = iqr = np.nan
            outlier_count = 0
        else:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outlier_count = int(((values < low) | (values > high)).sum())

        # Temporal variation: mean spatial standard deviation across nodes.
        temporal_std_per_node = np.nanstd(arr, axis=0)
        mean_node_temporal_std = float(np.nanmean(temporal_std_per_node))
        spatial_mean_series = np.nanmean(arr, axis=1)
        temporal_std_spatial_mean = float(np.nanstd(spatial_mean_series))

        rows.append({
            "feature": name,
            "min": float(np.nanmin(values)) if values.size else np.nan,
            "max": float(np.nanmax(values)) if values.size else np.nan,
            "mean": float(np.nanmean(values)) if values.size else np.nan,
            "std": float(np.nanstd(values)) if values.size else np.nan,
            "missing_pct": float(100.0 * (~finite).mean()),
            "outlier_count_iqr": outlier_count,
            "outlier_pct_iqr": float(100.0 * outlier_count / values.size) if values.size else np.nan,
            "temporal_std_spatial_mean": temporal_std_spatial_mean,
            "mean_node_temporal_std": mean_node_temporal_std,
        })

    # Coordinates are static.
    lat = np.asarray(ds["latitude"].values[lat_idx], dtype=float)
    lon = np.asarray(ds["longitude"].values[lon_idx], dtype=float)
    for name, values in (("latitude", lat), ("longitude", lon)):
        rows.append({
            "feature": name,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "missing_pct": 0.0,
            "outlier_count_iqr": 0,
            "outlier_pct_iqr": 0.0,
            "temporal_std_spatial_mean": 0.0,
            "mean_node_temporal_std": 0.0,
        })

    doy = times.dayofyear.to_numpy(dtype=float)
    sin_values = np.sin(2.0 * np.pi * doy / 365.25)
    cos_values = np.cos(2.0 * np.pi * doy / 365.25)
    for name, values in (("sin_day_of_year", sin_values), ("cos_day_of_year", cos_values)):
        rows.append({
            "feature": name,
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "missing_pct": 0.0,
            "outlier_count_iqr": 0,
            "outlier_pct_iqr": 0.0,
            "temporal_std_spatial_mean": float(values.std()),
            "mean_node_temporal_std": float(values.std()),
        })

    stats = pd.DataFrame(rows)
    stats["feature_order_index"] = [FEATURE_ORDER.index(x) for x in stats["feature"]]
    stats = stats.sort_values("feature_order_index").drop(columns="feature_order_index")
    stats.to_csv(REPORT_DIR / "feature_statistics.csv", index=False)
    return stats


# -----------------------------------------------------------------------------
# Build full feature/target arrays
# -----------------------------------------------------------------------------

def build_full_arrays(ds: xr.Dataset, lat_idx, lon_idx):
    times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    n_nodes = len(lat_idx)

    dynamic = {
        name: extract_time_space(ds[name], lat_idx, lon_idx)
        for name in TARGET_ORDER
    }

    bath = np.asarray(ds["bathymetry"].values[lat_idx, lon_idx], dtype=np.float64)
    bath = np.repeat(bath[None, :], len(times), axis=0)

    lat = np.asarray(ds["latitude"].values[lat_idx], dtype=np.float64)
    lon = np.asarray(ds["longitude"].values[lon_idx], dtype=np.float64)
    lat = np.repeat(lat[None, :], len(times), axis=0)
    lon = np.repeat(lon[None, :], len(times), axis=0)

    sin_doy, cos_doy = temporal_features(times, n_nodes)

    components = {
        **dynamic,
        "bathymetry": bath,
        "latitude": lat,
        "longitude": lon,
        "sin_day_of_year": sin_doy,
        "cos_day_of_year": cos_doy,
    }

    X = np.stack([components[name] for name in FEATURE_ORDER], axis=-1)
    Y = np.stack([dynamic[name] for name in TARGET_ORDER], axis=-1)

    if X.ndim != 3 or Y.ndim != 3:
        raise RuntimeError(f"Unexpected array dimensions: X={X.shape}, Y={Y.shape}")

    return X, Y, times


# -----------------------------------------------------------------------------
# Chronological split and scaling
# -----------------------------------------------------------------------------

def sequence_target_indices(n_days: int) -> np.ndarray:
    """Return raw time index for each t+1 target."""
    return np.arange(HISTORY_DAYS, n_days - FORECAST_HORIZON_DAYS + 1)


def split_sequence_indices(n_sequences: int):
    if abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must equal 1.")

    train_end = int(n_sequences * TRAIN_FRAC)
    val_end = train_end + int(n_sequences * VAL_FRAC)

    if train_end <= 0 or val_end >= n_sequences:
        raise RuntimeError("Chronological split produced an empty split.")

    return train_end, val_end


def fit_scalers_training_only(X: np.ndarray, Y: np.ndarray, target_indices: np.ndarray, train_end: int):
    """
    Fit all learned scalers only from data that belongs to the training period.

    Training target indices correspond to Y samples. For X, all seven history days
    of the training samples are used. This means no validation/test date is used to
    fit a scaler.
    """
    train_target_indices = target_indices[:train_end]
    train_input_end = int(train_target_indices[-1])  # exclusive raw input index
    train_target_end = int(train_target_indices[-1]) + 1

    X_scaled = X.copy()
    Y_scaled = Y.copy()

    feature_scalers = {}
    target_scalers = {}

    # Coordinates are deterministic and scaled using the fixed spatial range.
    for j, name in enumerate(FEATURE_ORDER):
        channel = X[:, :, j]

        if name == "latitude":
            mn, mx = LAT_MIN, LAT_MAX
            X_scaled[:, :, j] = ((channel - mn) / (mx - mn)) * 2.0 - 1.0
            feature_scalers[name] = {"type": "minmax_fixed", "min": mn, "max": mx}
            continue

        if name == "longitude":
            mn, mx = LON_MIN, LON_MAX
            X_scaled[:, :, j] = ((channel - mn) / (mx - mn)) * 2.0 - 1.0
            feature_scalers[name] = {"type": "minmax_fixed", "min": mn, "max": mx}
            continue

        if name in {"sin_day_of_year", "cos_day_of_year"}:
            # These are already bounded deterministic features. Keep them unchanged.
            feature_scalers[name] = {"type": "identity", "min": -1.0, "max": 1.0}
            continue

        # Fit using the raw time indices that appear in training input windows.
        train_values = channel[:train_input_end].reshape(-1, 1)
        train_values = train_values[np.isfinite(train_values[:, 0])]
        if train_values.size == 0:
            raise RuntimeError(f"No finite training values available for feature {name}")

        scaler = StandardScaler()
        scaler.fit(train_values.reshape(-1, 1))

        flat = channel.reshape(-1, 1)
        valid = np.isfinite(flat[:, 0])
        flat_scaled = flat.copy()
        flat_scaled[valid, 0] = scaler.transform(flat[valid].reshape(-1, 1)).ravel()
        X_scaled[:, :, j] = flat_scaled.reshape(channel.shape)
        feature_scalers[name] = {
            "type": "standard",
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
            "n_training_values": int(scaler.n_samples_seen_),
        }

    # Targets: fit using training-period raw target dates only.
    for j, name in enumerate(TARGET_ORDER):
        train_values = Y[:train_end, :, j].reshape(-1, 1)
        train_values = train_values[np.isfinite(train_values[:, 0])]
        if train_values.size == 0:
            raise RuntimeError(f"No finite training values available for target {name}")

        scaler = StandardScaler()
        scaler.fit(train_values.reshape(-1, 1))

        flat = Y[:, :, j].reshape(-1, 1)
        valid = np.isfinite(flat[:, 0])
        flat_scaled = flat.copy()
        flat_scaled[valid, 0] = scaler.transform(flat[valid].reshape(-1, 1)).ravel()
        Y_scaled[:, :, j] = flat_scaled.reshape(Y[:, :, j].shape)
        target_scalers[name] = {
            "type": "standard",
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
            "n_training_values": int(scaler.n_samples_seen_),
        }

    scalers = {
        "feature_order": FEATURE_ORDER,
        "target_order": TARGET_ORDER,
        "features": feature_scalers,
        "targets": target_scalers,
        "training_only": True,
        "note": "StandardScaler statistics are fitted only from chronological training-period values.",
    }

    return X_scaled, Y_scaled, scalers


# -----------------------------------------------------------------------------
# Sliding windows
# -----------------------------------------------------------------------------

def build_sequences(X_scaled, Y_scaled, times):
    """Build t-6...t -> t+1 sequences."""
    n_days = len(times)
    target_indices = sequence_target_indices(n_days)

    X_list = []
    Y_list = []
    target_dates = []
    target_raw_indices = []

    for target_idx in target_indices:
        input_start = target_idx - HISTORY_DAYS
        input_end = target_idx

        x_window = X_scaled[input_start:input_end]
        y_target = Y_scaled[target_idx]

        if x_window.shape[0] != HISTORY_DAYS:
            continue
        if not np.isfinite(x_window).all() or not np.isfinite(y_target).all():
            # This should normally not occur because fixed nodes were selected after
            # cleaning. If it does, fail rather than silently corrupting the dataset.
            raise RuntimeError(
                f"Non-finite value found in sequence targeting {times[target_idx].date()}"
            )

        X_list.append(x_window)
        Y_list.append(y_target)
        target_dates.append(pd.Timestamp(times[target_idx]))
        target_raw_indices.append(int(target_idx))

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    target_dates = pd.DatetimeIndex(target_dates)
    target_raw_indices = np.asarray(target_raw_indices, dtype=int)

    return X, Y, target_dates, target_raw_indices


def chronological_split(X, Y, target_dates, target_raw_indices):
    n = len(target_dates)
    train_end, val_end = split_sequence_indices(n)

    splits = {
        "train": (X[:train_end], Y[:train_end], target_dates[:train_end], target_raw_indices[:train_end]),
        "val": (X[train_end:val_end], Y[train_end:val_end], target_dates[train_end:val_end], target_raw_indices[train_end:val_end]),
        "test": (X[val_end:], Y[val_end:], target_dates[val_end:], target_raw_indices[val_end:]),
    }

    # Explicit future leakage checks.
    if not splits["train"][2].max() < splits["val"][2].min():
        raise RuntimeError("Training and validation target dates overlap.")
    if not splits["val"][2].max() < splits["test"][2].min():
        raise RuntimeError("Validation and test target dates overlap.")

    return splits


# -----------------------------------------------------------------------------
# Variability and augmentation decision
# -----------------------------------------------------------------------------

def variability_check(X_train, Y_train) -> str:
    print("\n[7/9] Checking post-preprocessing variability...")
    rows = []
    decision = "NO_AUGMENTATION"

    for j, name in enumerate(TARGET_ORDER):
        values = Y_train[:, :, j]
        std = float(np.std(values))
        mean = float(np.mean(values))
        useful = bool(std > MIN_USEFUL_STD)
        rows.append({
            "variable": name,
            "post_preprocessing_mean": mean,
            "post_preprocessing_std": std,
            "useful_variation": useful,
        })

    df = pd.DataFrame(rows)
    if not bool(df["useful_variation"].all()):
        decision = "CONTROLLED_AUGMENTATION_REVIEW_REQUIRED"

    df.to_csv(REPORT_DIR / "post_preprocessing_variability.csv", index=False)

    augmentation = {
        "decision": decision,
        "applied_to_production_arrays": False,
        "reason": "Production data is never automatically augmented. A separate controlled experiment is required if useful variation is insufficient.",
        "threshold_std": MIN_USEFUL_STD,
        "candidate_experiment": {
            "method": "physics-safe perturbation only after approval",
            "principle": "preserve observed physical bounds and temporal structure",
            "baseline": "untouched training data",
        },
    }
    save_json(REPORT_DIR / "augmentation_decision.json", augmentation)
    return decision


# -----------------------------------------------------------------------------
# Spatial kNN graph
# -----------------------------------------------------------------------------

def build_graph(node_lat: np.ndarray, node_lon: np.ndarray):
    print("\n[8/9] Building spatial kNN graph...")

    coords_deg = np.column_stack([node_lat, node_lon]).astype(np.float64)
    n_nodes = len(coords_deg)
    if n_nodes <= K_NEIGHBORS:
        raise RuntimeError(f"Only {n_nodes} nodes available; k={K_NEIGHBORS} is too large.")

    # NearestNeighbors with Euclidean lat/lon distances is acceptable over this
    # 10-degree Bay of Bengal regional domain. We retain degree coordinates for
    # reproducibility and save the graph distances as the same coordinate metric.
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm="ball_tree", metric="euclidean")
    knn.fit(coords_deg)
    distances, indices = knn.kneighbors(coords_deg)

    # Remove self-neighbour.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    src = np.repeat(np.arange(n_nodes), K_NEIGHBORS)
    dst = indices.reshape(-1)
    edge_distance = distances.reshape(-1)

    # Make the graph symmetric so message passing is not dependent on neighbour
    # direction. Duplicate edges are removed.
    pairs = set()
    symmetric_src = []
    symmetric_dst = []
    symmetric_dist = []

    for s, d, dist in zip(src, dst, edge_distance):
        for a, b in ((int(s), int(d)), (int(d), int(s))):
            if a == b:
                continue
            key = (a, b)
            if key not in pairs:
                pairs.add(key)
                symmetric_src.append(a)
                symmetric_dst.append(b)
                symmetric_dist.append(float(dist))

    edge_index = np.asarray([symmetric_src, symmetric_dst], dtype=np.int64)
    edge_distance = np.asarray(symmetric_dist, dtype=np.float32)

    graph = {
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_attr": torch.from_numpy(edge_distance[:, None]).float(),
        "edge_weight": torch.from_numpy((1.0 / (edge_distance + 1e-6))).float(),
        "num_nodes": int(n_nodes),
        "k": int(K_NEIGHBORS),
        "node_lat": torch.from_numpy(node_lat.astype(np.float32)),
        "node_lon": torch.from_numpy(node_lon.astype(np.float32)),
        "coordinate_order": "same N ordering as X/Y",
        "distance_metric": "euclidean degrees on latitude/longitude regional grid",
    }

    torch.save(graph, GRAPH_DIR / "graph.pt")

    coordinates = np.column_stack([node_lat, node_lon]).astype(np.float32)
    np.save(FINAL_DIR / "coordinates.npy", coordinates)
    np.save(FINAL_DIR / "node_lat.npy", node_lat.astype(np.float32))
    np.save(FINAL_DIR / "node_lon.npy", node_lon.astype(np.float32))

    return graph


# -----------------------------------------------------------------------------
# Metadata, README and final verification
# -----------------------------------------------------------------------------

def package_metadata(ds, node_mask, splits, scalers, common_start, common_end, augmentation_decision, graph):
    n_nodes = int(node_mask.sum())

    meta = {
        "project": "PIGT - Physics-Informed Graph Transformer",
        "region": {
            "latitude_min": LAT_MIN,
            "latitude_max": LAT_MAX,
            "longitude_min": LON_MIN,
            "longitude_max": LON_MAX,
        },
        "source_files": {key: str(path.relative_to(ROOT)) for key, path in RAW_FILES.items()},
        "common_period": {
            "start": str(common_start.date()),
            "end": str(common_end.date()),
        },
        "temporal_resolution": "daily",
        "history_window": "t-6 ... t",
        "forecast_horizon": "t+1",
        "history_days": HISTORY_DAYS,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "n_input_features": len(FEATURE_ORDER),
        "n_target_variables": len(TARGET_ORDER),
        "feature_order": FEATURE_ORDER,
        "target_order": TARGET_ORDER,
        "grid_shape": list(node_mask.shape),
        "n_ocean_nodes": n_nodes,
        "expected_shapes": {
            "X": ["B", HISTORY_DAYS, n_nodes, len(FEATURE_ORDER)],
            "Y": ["B", n_nodes, len(TARGET_ORDER)],
        },
        "split": {
            name: {
                "count": int(len(values[2])),
                "target_start": str(values[2][0].date()),
                "target_end": str(values[2][-1].date()),
            }
            for name, values in splits.items()
        },
        "normalization": {
            "feature_scaler_file": "feature_scaler.pkl",
            "training_only": True,
            "method": "StandardScaler for learned numerical channels; fixed [-1,1] scaling for latitude/longitude; identity for sine/cosine time channels.",
            "target_scalers_saved": True,
        },
        "cleaning": {
            "max_short_gap_days": MAX_GAP_DAYS,
            "land_handling": "land cells excluded by bathymetry-derived ocean mask",
            "persistent_missing_handling": "nodes with persistent missing values in any target variable excluded",
            "bathymetry_variable": "deptho",
        },
        "graph": {
            "file": "../graph/graph.pt",
            "k": K_NEIGHBORS,
            "num_nodes": n_nodes,
            "num_edges": int(graph["edge_index"].shape[1]),
            "coordinates": "coordinates.npy",
        },
        "augmentation": {
            "decision": augmentation_decision,
            "applied_to_final_arrays": False,
        },
        "files": {
            "X_train": "X_train.npy",
            "X_val": "X_val.npy",
            "X_test": "X_test.npy",
            "Y_train": "Y_train.npy",
            "Y_val": "Y_val.npy",
            "Y_test": "Y_test.npy",
            "scaler": "feature_scaler.pkl",
            "metadata": "metadata.json",
            "coordinates": "coordinates.npy",
            "node_mask": "node_mask.npy",
        },
    }

    save_json(FINAL_DIR / "metadata.json", meta)
    return meta


def write_readme(meta):
    text = f"""# PIGT Final Dataset\n\n## Purpose\nTraining-ready daily ocean-environment dataset for the PIGT models.\n\n## Shapes\n- `X_train`, `X_val`, `X_test`: `[B, 7, N, 13]`\n- `Y_train`, `Y_val`, `Y_test`: `[B, N, 8]`\n- `N = {meta['n_ocean_nodes']}` ocean nodes\n\n## Input feature order\n```text\n{chr(10).join(f'{i}: {name}' for i, name in enumerate(FEATURE_ORDER))}\n```\n\n## Target order\n```text\n{chr(10).join(f'{i}: {name}' for i, name in enumerate(TARGET_ORDER))}\n```\n\n## Time protocol\nSeven historical daily steps `t-6 ... t` are used to predict the next day `t+1`.\nThe splits are chronological 70/15/15 by target date. Future validation/test periods are not used to fit the scalers.\n\n## Raw data alignment\n- Copernicus SST, salinity and surface currents are already daily on the 121x121 ocean grid.\n- Copernicus Wave `VHM0` is aggregated from 3-hourly to daily and interpolated to the ocean grid.\n- ERA5 `u10`, `v10`, and `msl` are aggregated from 6-hourly to daily and interpolated to the ocean grid.\n- Wind speed and meteorological wind direction are derived from `u10`/`v10`.\n- ERA5 pressure is stored in hPa.\n- Bathymetry uses the supplied `deptho` variable; finite positive `deptho` defines the ocean mask.\n\n## Cleaning\nOnly temporal gaps of up to {MAX_GAP_DAYS} consecutive days are linearly interpolated. Land cells are never filled numerically. Nodes that remain incomplete in any target variable after cleaning are excluded so every final sample has a fixed N.\n\n## Files\n- `X_train.npy`, `X_val.npy`, `X_test.npy`\n- `Y_train.npy`, `Y_val.npy`, `Y_test.npy`\n- `feature_scaler.pkl` — feature and target scaling metadata\n- `metadata.json` — complete dataset contract, feature order, target order, split dates and graph metadata\n- `coordinates.npy`, `node_lat.npy`, `node_lon.npy` — graph node coordinates in exactly the same N order as X/Y\n- `node_mask.npy` — full 121x121 final ocean-node mask\n- Graph: `data/graph/graph.pt`\n\n## Load example\n```python\nimport pickle\nfrom pathlib import Path\nimport numpy as np\nimport torch\n\nfinal = Path('data/final')\nX_train = np.load(final / 'X_train.npy')\nY_train = np.load(final / 'Y_train.npy')\n\nwith open(final / 'feature_scaler.pkl', 'rb') as f:\n    scalers = pickle.load(f)\n\ngraph = torch.load('data/graph/graph.pt', map_location='cpu')\n\nprint('X_train:', X_train.shape)\nprint('Y_train:', Y_train.shape)\nprint('edges:', graph['edge_index'].shape)\n```\n\n## Augmentation\nDecision: `{meta['augmentation']['decision']}`. No augmentation is applied automatically. Any augmentation must be a separate controlled experiment and compared against the untouched baseline.\n"""
    (FINAL_DIR / "README.md").write_text(text, encoding="utf-8")


def verify_final_package(meta):
    print("\n[9/9] Verifying final package...")
    failures = []

    expected_suffixes = {
        "X_train.npy": (HISTORY_DAYS, meta["n_ocean_nodes"], len(FEATURE_ORDER)),
        "X_val.npy": (HISTORY_DAYS, meta["n_ocean_nodes"], len(FEATURE_ORDER)),
        "X_test.npy": (HISTORY_DAYS, meta["n_ocean_nodes"], len(FEATURE_ORDER)),
        "Y_train.npy": (meta["n_ocean_nodes"], len(TARGET_ORDER)),
        "Y_val.npy": (meta["n_ocean_nodes"], len(TARGET_ORDER)),
        "Y_test.npy": (meta["n_ocean_nodes"], len(TARGET_ORDER)),
    }

    for filename, suffix in expected_suffixes.items():
        path = FINAL_DIR / filename
        if not path.exists():
            failures.append(f"missing {filename}")
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.shape[1:] != suffix:
            failures.append(f"{filename}: shape {arr.shape}, expected (*,{suffix})")
        if not np.isfinite(arr).all():
            failures.append(f"{filename}: contains NaN/Inf")

    required_files = [
        FINAL_DIR / "feature_scaler.pkl",
        FINAL_DIR / "metadata.json",
        FINAL_DIR / "README.md",
        FINAL_DIR / "coordinates.npy",
        FINAL_DIR / "node_lat.npy",
        FINAL_DIR / "node_lon.npy",
        FINAL_DIR / "node_mask.npy",
        GRAPH_DIR / "graph.pt",
    ]
    for path in required_files:
        if not path.exists():
            failures.append(f"missing {path}")

    graph = torch.load(GRAPH_DIR / "graph.pt", map_location="cpu", weights_only=False)
    if int(graph["num_nodes"]) != meta["n_ocean_nodes"]:
        failures.append("graph node count does not match X/Y N")
    if graph["edge_index"].shape[0] != 2:
        failures.append("graph edge_index must have shape [2,E]")

    if meta["feature_order"] != FEATURE_ORDER:
        failures.append("feature order mismatch")
    if meta["target_order"] != TARGET_ORDER:
        failures.append("target order mismatch")
    if len(TARGET_ORDER) != 8:
        failures.append("target order is not 8 variables")

    if failures:
        raise RuntimeError("FINAL PACKAGE VERIFICATION FAILED:\n- " + "\n- ".join(failures))

    print("  FINAL PACKAGE VERIFICATION: PASS")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PIGT — COMPLETE OCEAN DATA PREPARATION PIPELINE")
    print("=" * 80)

    # 1. Audit
    audit_all_raw()

    # 2. Align and clean
    ds, common_start, common_end = load_and_align()

    # 3. Invalid-cell audit
    invalid_cell_audit(ds)

    # 4. Fixed ocean nodes
    lat_idx, lon_idx, node_mask = build_common_ocean_nodes(ds)
    node_lat = np.asarray(ds["latitude"].values[lat_idx], dtype=np.float64)
    node_lon = np.asarray(ds["longitude"].values[lon_idx], dtype=np.float64)

    # Save full-grid coordinates/mask.
    np.save(FINAL_DIR / "lat.npy", np.asarray(ds["latitude"].values, dtype=np.float32))
    np.save(FINAL_DIR / "lon.npy", np.asarray(ds["longitude"].values, dtype=np.float32))
    np.save(FINAL_DIR / "node_mask.npy", node_mask.astype(bool))

    # 5. Feature statistics
    times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    calculate_feature_statistics(ds, lat_idx, lon_idx, times)

    # 6. Build arrays and sequences
    print("\n[6/9] Building 13-feature / 8-target arrays and sliding windows...")
    X_raw, Y_raw, times = build_full_arrays(ds, lat_idx, lon_idx)

    n_sequences = len(times) - HISTORY_DAYS - FORECAST_HORIZON_DAYS + 1
    if n_sequences <= 0:
        raise RuntimeError("Not enough daily observations for a 7-day history + 1-day horizon.")

    target_indices = sequence_target_indices(len(times))
    train_end, _ = split_sequence_indices(n_sequences)

    # Fit scalers BEFORE creating validation/test samples.
    X_scaled, Y_scaled, scalers = fit_scalers_training_only(
        X_raw,
        Y_raw,
        target_indices,
        train_end,
    )

    with (FINAL_DIR / "feature_scaler.pkl").open("wb") as f:
        pickle.dump(scalers, f)

    X, Y, target_dates, target_raw_indices = build_sequences(
        X_scaled,
        Y_scaled,
        times,
    )

    splits = chronological_split(X, Y, target_dates, target_raw_indices)

    for name, (x, y, dates, _) in splits.items():
        np.save(FINAL_DIR / f"X_{name}.npy", x)
        np.save(FINAL_DIR / f"Y_{name}.npy", y)
        print(
            f"  {name.upper():5s}: X={x.shape}, Y={y.shape}, "
            f"target dates={dates[0].date()} -> {dates[-1].date()}"
        )

    # 7. Variability / augmentation decision
    augmentation_decision = variability_check(
        splits["train"][0],
        splits["train"][1],
    )

    # 8. Graph
    graph = build_graph(node_lat, node_lon)

    # Metadata and README
    meta = package_metadata(
        ds,
        node_mask,
        splits,
        scalers,
        common_start,
        common_end,
        augmentation_decision,
        graph,
    )
    write_readme(meta)

    # Final verification
    verify_final_package(meta)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Final dataset : {FINAL_DIR}")
    print(f"Graph         : {GRAPH_DIR / 'graph.pt'}")
    print(f"Reports       : {REPORT_DIR}")
    print(f"N nodes       : {meta['n_ocean_nodes']}")
    print("X shape       : [B, 7, N, 13]")
    print("Y shape       : [B, N, 8]")


if __name__ == "__main__":
    main()
