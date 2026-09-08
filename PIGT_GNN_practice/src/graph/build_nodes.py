"""
STEP 1 of 8 — build_nodes.py
====================================================================
Turns a CMEMS NetCDF grid into a flat table: one row per (node, time).
A "node" here is one lat/lon grid cell.

For a first practice run this uses SYNTHETIC data shaped exactly like
GLOBAL_ANALYSISFORECAST_PHY_001_024 (thetao, so, zos, uo, vo).

To use REAL data instead:
  1. pip install copernicusmarine
  2. copernicusmarine login   (one-time, free account at data.marine.copernicus.eu)
  3. Run the copernicusmarine.subset(...) call in fetch_real_data() below
  4. Set USE_SYNTHETIC = False and point RAW_NC_PATH at the downloaded file

Output: data/processed_nodes.csv
    columns: node_id, lat, lon, time, thetao, so, zos, uo, vo
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

USE_SYNTHETIC = False
RAW_NC_PATH = "data/raw/cmems_subset.nc"
OUTPUT_PATH = "data/processed_nodes.csv"

LAT_MIN, LAT_MAX = 10.0, 20.0
LON_MIN, LON_MAX = 80.0, 90.0
VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


def fetch_real_data():
    """Run this once, on your own machine, after `copernicusmarine login`."""
    import copernicusmarine
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",  # verify on the product's Data Access tab
        variables=VARIABLES,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-06-30T00:00:00",
        minimum_depth=0, maximum_depth=1,
        output_filename="cmems_subset.nc",
        output_directory="data/raw",
    )


def make_synthetic_dataset(n_time=60, n_lat=20, n_lon=20, seed=1):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=n_time, freq="D")
    lat = np.linspace(LAT_MIN, LAT_MAX, n_lat)
    lon = np.linspace(LON_MIN, LON_MAX, n_lon)
    t_idx = np.arange(n_time)[:, None, None]
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")

    thetao = 27 + 2 * np.sin(2 * np.pi * t_idx / 365) + 0.05 * lat_g + rng.normal(0, 0.3, (n_time, n_lat, n_lon))
    so = 35 + 0.5 * np.cos(2 * np.pi * t_idx / 365) + rng.normal(0, 0.1, (n_time, n_lat, n_lon))
    zos = 0.1 * np.sin(2 * np.pi * t_idx / 180 + lon_g / 10) + rng.normal(0, 0.02, (n_time, n_lat, n_lon))
    uo = 0.2 * np.sin(2 * np.pi * t_idx / 90) + rng.normal(0, 0.05, (n_time, n_lat, n_lon))
    vo = 0.2 * np.cos(2 * np.pi * t_idx / 90) + rng.normal(0, 0.05, (n_time, n_lat, n_lon))

    mask = rng.random((n_time, n_lat, n_lon)) < 0.02
    for arr in (thetao, so, zos, uo, vo):
        arr[mask] = np.nan

    return xr.Dataset(
        {
            "thetao": (("time", "latitude", "longitude"), thetao),
            "so": (("time", "latitude", "longitude"), so),
            "zos": (("time", "latitude", "longitude"), zos),
            "uo": (("time", "latitude", "longitude"), uo),
            "vo": (("time", "latitude", "longitude"), vo),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def main():
    if USE_SYNTHETIC:
        print("[INFO] Using synthetic dataset for first practice run.")
        ds = make_synthetic_dataset()
    else:
        print(f"[INFO] Loading real CMEMS file: {RAW_NC_PATH}")
        ds = xr.open_dataset(RAW_NC_PATH)
        ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX))

    ds = ds[VARIABLES].interpolate_na(dim="time", method="linear").ffill(dim="time").bfill(dim="time")

    # Cells that are STILL NaN after time-interpolation were NaN at every single
    # timestep -- that means land, not a data gap. Interpolation across time can't
    # fix a cell with zero valid days to interpolate from. Drop these before the
    # graph is built, or they poison every neighbor's prediction via message passing.
    still_missing = ds[VARIABLES[0]].isnull()
    for var in VARIABLES[1:]:
        still_missing = still_missing | ds[var].isnull()
    ocean_mask_2d = ~still_missing.any(dim="time")   # True = valid ocean cell at every timestep

    lat_g, lon_g = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    full_node_id = np.arange(lat_g.size)
    ocean_flat_mask = ocean_mask_2d.values.ravel()

    # remap surviving ocean nodes to a contiguous 0..N-1 range (required downstream
    # by create_pyg_graph.py's node_id == range(N) check)
    ocean_old_ids = full_node_id[ocean_flat_mask]
    remap = {old: new for new, old in enumerate(ocean_old_ids)}

    n_dropped = int((~ocean_flat_mask).sum())
    n_kept = int(ocean_flat_mask.sum())
    print(f"[INFO] Dropping {n_dropped} land / permanently-missing grid cells; "
          f"keeping {n_kept} ocean nodes.")

    rows = []
    for t_i, t in enumerate(ds.time.values):
        frame = pd.DataFrame({
            "node_id": full_node_id,
            "lat": lat_g.ravel(),
            "lon": lon_g.ravel(),
            "time": t,
        })
        for var in VARIABLES:
            frame[var] = ds[var].isel(time=t_i).values.ravel()
        frame = frame[ocean_flat_mask].copy()
        frame["node_id"] = frame["node_id"].map(remap)
        rows.append(frame)

    node_table = pd.concat(rows, ignore_index=True)
    Path("data").mkdir(exist_ok=True)
    node_table.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved {len(node_table)} rows ({n_kept} nodes x {len(ds.time)} timesteps) "
          f"to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()