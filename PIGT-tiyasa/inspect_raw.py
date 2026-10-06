"""
PIGT Ocean Dataset - Raw Data Inspection

Purpose:
    Inspect all raw NetCDF files before running the main data pipeline.

Checks:
    - File existence
    - Dimensions
    - Coordinates
    - Variables
    - Units
    - Time range
    - Number of timesteps
    - Duplicate timestamps
    - Time-step spacing
    - Latitude / longitude coverage
    - Depth levels
    - NaN / missing values
    - Infinite values
    - Basic coordinate validity
    - Expected variables
    - Expected Bay of Bengal spatial range

Expected project region:
    Latitude  : 10°N to 20°N
    Longitude : 80°E to 90°E

Run:
    python inspect_raw.py

Output:
    results/
        raw_inspection_report.json
        raw_variable_statistics.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED FILES
# ============================================================

FILES = {
    "copernicus_physics_thetao": (
        RAW_DIR / "copernicus_physics" / "thetao_bob.nc"
    ),
    "copernicus_physics_salinity": (
        RAW_DIR / "copernicus_physics" / "so_bob.nc"
    ),
    "copernicus_physics_currents": (
        RAW_DIR / "copernicus_physics" / "currents_bob.nc"
    ),
    "copernicus_physics_bathymetry": (
        RAW_DIR / "copernicus_physics" / "bathymetry.nc"
    ),
    "copernicus_wave": (
        RAW_DIR / "copernicus_wave" / "vhm0_bob.nc"
    ),
    "era5": (
        RAW_DIR / "era5" / "era5_bob.nc"
    ),
}


# ============================================================
# EXPECTED VARIABLES FROM PROJECT GUIDE
# ============================================================

EXPECTED_VARIABLES = {
    "copernicus_physics_thetao": ["thetao"],
    "copernicus_physics_salinity": ["so"],
    "copernicus_physics_currents": ["uo", "vo"],
    "copernicus_physics_bathymetry": ["depth", "mask"],
    "copernicus_wave": ["VHM0"],
    "era5": ["u10", "v10", "msl"],
}


# ============================================================
# EXPECTED BAY OF BENGAL REGION
# ============================================================

LAT_MIN = 10.0
LAT_MAX = 20.0

LON_MIN = 80.0
LON_MAX = 90.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_coordinate(ds, candidates):
    """Find the first matching coordinate/dimension name."""
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name

    return None


def get_time_info(ds):
    """Inspect time coordinate."""

    time_name = find_coordinate(
        ds,
        [
            "time",
            "valid_time",
            "datetime",
            "date",
        ],
    )

    if time_name is None:
        return {
            "time_coordinate": None,
            "time_count": None,
            "start": None,
            "end": None,
            "duplicate_timestamps": None,
            "median_step_hours": None,
            "unique_steps_hours": [],
        }

    values = ds[time_name].values

    try:
        times = pd.to_datetime(values)
    except Exception:
        return {
            "time_coordinate": time_name,
            "time_count": len(values),
            "start": str(values[0]) if len(values) else None,
            "end": str(values[-1]) if len(values) else None,
            "duplicate_timestamps": None,
            "median_step_hours": None,
            "unique_steps_hours": [],
        }

    if len(times) == 0:
        return {
            "time_coordinate": time_name,
            "time_count": 0,
            "start": None,
            "end": None,
            "duplicate_timestamps": 0,
            "median_step_hours": None,
            "unique_steps_hours": [],
        }

    duplicate_count = int(times.duplicated().sum())

    if len(times) > 1:
        differences = pd.Series(times).diff().dropna()

        step_hours = (
            differences.dt.total_seconds() / 3600.0
        )

        unique_steps = sorted(
            set(np.round(step_hours.values, 6).tolist())
        )

        median_step = float(np.median(step_hours.values))

    else:
        unique_steps = []
        median_step = None

    return {
        "time_coordinate": time_name,
        "time_count": int(len(times)),
        "start": str(times.min()),
        "end": str(times.max()),
        "duplicate_timestamps": duplicate_count,
        "median_step_hours": median_step,
        "unique_steps_hours": unique_steps[:20],
    }


def get_spatial_info(ds):
    """Inspect latitude and longitude."""

    lat_name = find_coordinate(
        ds,
        [
            "latitude",
            "lat",
            "nav_lat",
            "y",
        ],
    )

    lon_name = find_coordinate(
        ds,
        [
            "longitude",
            "lon",
            "nav_lon",
            "x",
        ],
    )

    result = {
        "latitude_coordinate": lat_name,
        "longitude_coordinate": lon_name,
        "latitude_min": None,
        "latitude_max": None,
        "longitude_min": None,
        "longitude_max": None,
        "latitude_count": None,
        "longitude_count": None,
        "inside_expected_region": None,
    }

    if lat_name is not None:

        lat = np.asarray(ds[lat_name].values)

        finite_lat = lat[np.isfinite(lat)]

        if finite_lat.size:
            result["latitude_min"] = float(finite_lat.min())
            result["latitude_max"] = float(finite_lat.max())
            result["latitude_count"] = int(finite_lat.size)

    if lon_name is not None:

        lon = np.asarray(ds[lon_name].values)

        finite_lon = lon[np.isfinite(lon)]

        if finite_lon.size:
            result["longitude_min"] = float(finite_lon.min())
            result["longitude_max"] = float(finite_lon.max())
            result["longitude_count"] = int(finite_lon.size)

    if (
        result["latitude_min"] is not None
        and result["latitude_max"] is not None
        and result["longitude_min"] is not None
        and result["longitude_max"] is not None
    ):

        lat_ok = (
            result["latitude_min"] <= LAT_MIN
            and result["latitude_max"] >= LAT_MAX
        )

        lon_ok = (
            result["longitude_min"] <= LON_MIN
            and result["longitude_max"] >= LON_MAX
        )

        result["inside_expected_region"] = bool(lat_ok and lon_ok)

    return result


def get_depth_info(ds):
    """Inspect depth coordinate if present."""

    depth_name = find_coordinate(
        ds,
        [
            "depth",
            "deptht",
            "lev",
            "level",
        ],
    )

    result = {
        "depth_coordinate": depth_name,
        "depth_count": None,
        "depth_min": None,
        "depth_max": None,
    }

    if depth_name is not None:

        values = np.asarray(ds[depth_name].values)

        finite_values = values[np.isfinite(values)]

        if finite_values.size:

            result["depth_count"] = int(finite_values.size)
            result["depth_min"] = float(finite_values.min())
            result["depth_max"] = float(finite_values.max())

    return result


def inspect_variable(ds, variable_name):
    """Calculate statistics for one variable."""

    if variable_name not in ds.data_vars:
        return {
            "variable": variable_name,
            "exists": False,
        }

    da = ds[variable_name]

    result = {
        "variable": variable_name,
        "exists": True,
        "dimensions": list(da.dims),
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "units": str(da.attrs.get("units", "not specified")),
        "long_name": str(
            da.attrs.get("long_name", "not specified")
        ),
        "standard_name": str(
            da.attrs.get("standard_name", "not specified")
        ),
        "total_cells": int(da.size),
    }

    # --------------------------------------------------------
    # Missing / infinite values
    # --------------------------------------------------------

    try:

        values = np.asarray(da.values)

        if np.issubdtype(values.dtype, np.number):

            finite_mask = np.isfinite(values)

            finite_values = values[finite_mask]

            nan_count = int(np.isnan(values).sum())
            inf_count = int(np.isinf(values).sum())

            result["nan_count"] = nan_count
            result["inf_count"] = inf_count

            if da.size > 0:
                result["missing_percent"] = (
                    nan_count / da.size * 100.0
                )

            if finite_values.size:

                result["min"] = float(np.min(finite_values))
                result["max"] = float(np.max(finite_values))
                result["mean"] = float(np.mean(finite_values))
                result["std"] = float(np.std(finite_values))

            else:

                result["min"] = None
                result["max"] = None
                result["mean"] = None
                result["std"] = None

        else:

            result["nan_count"] = None
            result["inf_count"] = None
            result["missing_percent"] = None
            result["min"] = None
            result["max"] = None
            result["mean"] = None
            result["std"] = None

    except Exception as exc:

        result["statistics_error"] = str(exc)

    return result


def check_expected_variables(dataset_key, ds):
    """Check whether expected variables exist."""

    expected = EXPECTED_VARIABLES.get(
        dataset_key,
        [],
    )

    actual = list(ds.data_vars)

    missing = [
        variable
        for variable in expected
        if variable not in actual
    ]

    found = [
        variable
        for variable in expected
        if variable in actual
    ]

    return {
        "expected": expected,
        "found": found,
        "missing": missing,
        "all_expected_present": len(missing) == 0,
        "all_dataset_variables": actual,
    }


def check_coordinate_validity(ds):
    """Check latitude and longitude for invalid values."""

    result = {
        "latitude_invalid_count": None,
        "longitude_invalid_count": None,
        "latitude_out_of_normal_range": None,
        "longitude_out_of_normal_range": None,
    }

    lat_name = find_coordinate(
        ds,
        ["latitude", "lat", "nav_lat", "y"],
    )

    lon_name = find_coordinate(
        ds,
        ["longitude", "lon", "nav_lon", "x"],
    )

    if lat_name is not None:

        lat = np.asarray(ds[lat_name].values)

        result["latitude_invalid_count"] = int(
            (~np.isfinite(lat)).sum()
        )

        finite_lat = lat[np.isfinite(lat)]

        if finite_lat.size:
            result["latitude_out_of_normal_range"] = bool(
                np.any(
                    (finite_lat < -90)
                    | (finite_lat > 90)
                )
            )

    if lon_name is not None:

        lon = np.asarray(ds[lon_name].values)

        result["longitude_invalid_count"] = int(
            (~np.isfinite(lon)).sum()
        )

        finite_lon = lon[np.isfinite(lon)]

        if finite_lon.size:
            result["longitude_out_of_normal_range"] = bool(
                np.any(
                    (finite_lon < -360)
                    | (finite_lon > 360)
                )
            )

    return result


# ============================================================
# MAIN INSPECTION
# ============================================================

def inspect_dataset(dataset_key, path):

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_key}")
    print("=" * 80)

    result = {
        "dataset": dataset_key,
        "file": str(path),
        "exists": path.exists(),
    }

    if not path.exists():

        print("ERROR: File not found.")
        return result

    try:

        print(f"Opening: {path}")

        ds = xr.open_dataset(path)

        # ----------------------------------------------------
        # Basic dataset information
        # ----------------------------------------------------

        result["dimensions"] = {
            name: int(size)
            for name, size in ds.sizes.items()
        }

        result["coordinates"] = list(ds.coords)
        result["data_variables"] = list(ds.data_vars)

        result["attributes"] = {
            key: str(value)
            for key, value in ds.attrs.items()
        }

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        time_info = get_time_info(ds)

        result["time"] = time_info

        # ----------------------------------------------------
        # Spatial
        # ----------------------------------------------------

        result["spatial"] = get_spatial_info(ds)

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        result["depth"] = get_depth_info(ds)

        # ----------------------------------------------------
        # Expected variables
        # ----------------------------------------------------

        result["expected_variables"] = (
            check_expected_variables(
                dataset_key,
                ds,
            )
        )

        # ----------------------------------------------------
        # Coordinate validity
        # ----------------------------------------------------

        result["coordinate_validity"] = (
            check_coordinate_validity(ds)
        )

        # ----------------------------------------------------
        # Variable statistics
        # ----------------------------------------------------

        variable_results = []

        for variable_name in ds.data_vars:

            print(
                f"Inspecting variable: {variable_name}"
            )

            variable_results.append(
                inspect_variable(
                    ds,
                    variable_name,
                )
            )

        result["variables"] = variable_results

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print("\nDimensions:")
        print(result["dimensions"])

        print("\nVariables:")
        print(result["data_variables"])

        print("\nTime:")
        print(result["time"])

        print("\nSpatial:")
        print(result["spatial"])

        print("\nDepth:")
        print(result["depth"])

        print("\nExpected variables:")
        print(result["expected_variables"])

        ds.close()

    except Exception as exc:

        result["open_error"] = str(exc)

        print("\nERROR while opening/inspecting file:")
        print(exc)

    return result


def main():

    print("\n")
    print("=" * 80)
    print("PIGT RAW DATA INSPECTION")
    print("=" * 80)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Raw data     : {RAW_DIR}")
    print(f"Results      : {RESULTS_DIR}")

    all_results = []

    for dataset_key, path in FILES.items():

        result = inspect_dataset(
            dataset_key,
            path,
        )

        all_results.append(result)

    # ========================================================
    # Save JSON report
    # ========================================================

    report = {
        "project": "PIGT - Bay of Bengal Multivariate Forecasting",
        "inspection": {
            "latitude_range": [
                LAT_MIN,
                LAT_MAX,
            ],
            "longitude_range": [
                LON_MIN,
                LON_MAX,
            ],
            "files_checked": len(FILES),
        },
        "datasets": all_results,
    }

    json_path = (
        RESULTS_DIR /
        "raw_inspection_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # ========================================================
    # Save variable CSV
    # ========================================================

    rows = []

    for dataset in all_results:

        dataset_name = dataset.get(
            "dataset",
            "",
        )

        for variable in dataset.get(
            "variables",
            [],
        ):

            row = {
                "dataset": dataset_name,
                **variable,
            }

            rows.append(row)

    if rows:

        df = pd.DataFrame(rows)

        csv_path = (
            RESULTS_DIR /
            "raw_variable_statistics.csv"
        )

        df.to_csv(
            csv_path,
            index=False,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print(f"\nJSON report:")
    print(json_path)

    if rows:
        print("\nCSV report:")
        print(
            RESULTS_DIR /
            "raw_variable_statistics.csv"
        )

    print("\nNext step:")
    print(
        "Review the report before running "
        "data_pipeline.py."
    )


if __name__ == "__main__":
    main()