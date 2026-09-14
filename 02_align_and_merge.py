"""
STEPS 6, 7, 8, 9 — Select variables, handle depth, align all three sources
onto one common daily timestamp + spatial grid, merge, and build the shared
ocean/land mask with short-gap interpolation for transient NaNs.

Output:
    data/processed/merged_bob_daily.nc

USAGE:
    python 02_align_and_merge.py
"""

import numpy as np
import xarray as xr

from config import (
    RAW_PHYSICS_DIR,
    RAW_WAVE_DIR,
    RAW_ERA5_DIR,
    MERGED_NC_PATH,
)


def load_ocean_grid():
    """Load ocean physics variables at the surface level."""

    print("Loading ocean physics variables (surface level only)...")

    # ---------------------------------------------------------
    # Load Copernicus ocean physics
    # ---------------------------------------------------------
    thetao_ds = xr.open_dataset(
        RAW_PHYSICS_DIR / "thetao_bob.nc"
    )

    so_ds = xr.open_dataset(
        RAW_PHYSICS_DIR / "so_bob.nc"
    )

    cur_ds = xr.open_dataset(
        RAW_PHYSICS_DIR / "currents_bob.nc"
    )

    bathy_ds = xr.open_dataset(
        RAW_PHYSICS_DIR / "bathymetry.nc"
    )

    # ---------------------------------------------------------
    # Surface variables
    # depth=0 is the only available depth level
    # (~0.494 m)
    # ---------------------------------------------------------
    sst = (
        thetao_ds["thetao"]
        .isel(depth=0)
        .rename("sst")
    )

    salinity = (
        so_ds["so"]
        .isel(depth=0)
        .rename("salinity")
    )

    u_current = (
        cur_ds["uo"]
        .isel(depth=0)
        .rename("u_current")
    )

    v_current = (
        cur_ds["vo"]
        .isel(depth=0)
        .rename("v_current")
    )

    # ---------------------------------------------------------
    # Bathymetry
    #
    # Valid depth value = ocean
    # NaN = land
    # ---------------------------------------------------------
    bathymetry = (
        bathy_ds["deptho"]
        .rename("bathymetry")
    )

    # ---------------------------------------------------------
    # Ocean / land mask
    #
    # 1 = ocean
    # 0 = land
    # ---------------------------------------------------------
    land_mask = xr.where(
        bathymetry.notnull(),
        1,
        0
    ).rename("land_mask")

    return (
        sst,
        salinity,
        u_current,
        v_current,
        bathymetry,
        land_mask,
    )


def load_wave(ocean_lat, ocean_lon):
    """Load SWH, convert to daily mean and align to ocean grid."""

    print("Loading and resampling wave data (SWH)...")

    wave_ds = xr.open_dataset(
        RAW_WAVE_DIR / "vhm0_bob.nc"
    )

    swh = wave_ds["VHM0"]

    # ---------------------------------------------------------
    # Convert wave data to daily mean
    # ---------------------------------------------------------
    swh_daily = swh.resample(
        time="1D"
    ).mean()

    # ---------------------------------------------------------
    # Regrid wave data if its grid differs
    # from the ocean grid
    # ---------------------------------------------------------
    if not (
        swh_daily.latitude.equals(ocean_lat)
        and
        swh_daily.longitude.equals(ocean_lon)
    ):
        print(
            "  Wave grid differs from ocean grid — "
            "regridding (linear interpolation)..."
        )

        swh_daily = swh_daily.interp(
            latitude=ocean_lat,
            longitude=ocean_lon,
            method="linear",
        )

    return swh_daily.rename("swh")


def load_era5(ocean_lat, ocean_lon):
    """Load ERA5 wind and pressure, convert to daily values,
    and regrid onto the ocean grid."""

    print("Loading and processing ERA5 wind/pressure...")

    era5_ds = xr.open_dataset(
        RAW_ERA5_DIR / "era5_bob.nc"
    )

    # ---------------------------------------------------------
    # Normalize possible ERA5 time coordinate name
    # ---------------------------------------------------------
    if (
        "valid_time" in era5_ds.coords
        and
        "time" not in era5_ds.coords
    ):
        era5_ds = era5_ds.rename(
            {"valid_time": "time"}
        )

    u10 = era5_ds["u10"]
    v10 = era5_ds["v10"]
    msl_pa = era5_ds["msl"]

    # ---------------------------------------------------------
    # Daily averages
    # ---------------------------------------------------------
    u10_daily = u10.resample(
        time="1D"
    ).mean()

    v10_daily = v10.resample(
        time="1D"
    ).mean()

    msl_daily = msl_pa.resample(
        time="1D"
    ).mean()

    # ---------------------------------------------------------
    # Wind speed
    # ---------------------------------------------------------
    wind_speed_daily = (
        u10_daily ** 2
        +
        v10_daily ** 2
    ) ** 0.5

    wind_speed_daily.name = "wind_speed"

    # ---------------------------------------------------------
    # Meteorological wind direction
    # Direction from which wind is blowing
    # ---------------------------------------------------------
    wind_dir_daily = (
        180
        +
        np.degrees(
            np.arctan2(
                u10_daily,
                v10_daily,
            )
        )
    ) % 360

    wind_dir_daily.name = "wind_direction"

    # ---------------------------------------------------------
    # Mean sea-level pressure
    # Pa -> hPa
    # ---------------------------------------------------------
    pressure_daily = (
        msl_daily / 100.0
    ).rename("pressure")

    # ---------------------------------------------------------
    # Regrid ERA5 to 0.0833 degree ocean grid
    # ---------------------------------------------------------
    print(
        "  Regridding ERA5 onto ocean grid "
        "(linear interpolation)..."
    )

    wind_speed_regridded = wind_speed_daily.interp(
        latitude=ocean_lat,
        longitude=ocean_lon,
        method="linear",
    )

    wind_dir_regridded = wind_dir_daily.interp(
        latitude=ocean_lat,
        longitude=ocean_lon,
        method="linear",
    )

    pressure_regridded = pressure_daily.interp(
        latitude=ocean_lat,
        longitude=ocean_lon,
        method="linear",
    )

    return (
        wind_speed_regridded,
        wind_dir_regridded,
        pressure_regridded,
    )


def fill_short_gaps(da, max_gap_days=3):
    """
    Fill short NaN gaps along time without requiring
    the bottleneck package.

    Only gaps surrounded by valid values are interpolated.
    Longer gaps and gaps at the beginning/end remain NaN.
    """

    max_gap_steps = max_gap_days

    # Linear interpolation across time.
    # limit controls the maximum number of consecutive
    # missing daily values that can be filled.
    filled = da.interpolate_na(
        dim="time",
        method="linear",
        limit=max_gap_steps,
    )

    return filled


def main():

    # =========================================================
    # 1. Load ocean physics
    # =========================================================
    (
        sst,
        salinity,
        u_current,
        v_current,
        bathymetry,
        land_mask,
    ) = load_ocean_grid()

    ocean_lat = sst.latitude
    ocean_lon = sst.longitude

    # =========================================================
    # 2. Load wave data
    # =========================================================
    swh = load_wave(
        ocean_lat,
        ocean_lon,
    )

    # =========================================================
    # 3. Load ERA5 data
    # =========================================================
    (
        wind_speed,
        wind_direction,
        pressure,
    ) = load_era5(
        ocean_lat,
        ocean_lon,
    )

    # =========================================================
    # 4. Merge all variables
    # =========================================================
    print(
        "Merging all variables onto the common ocean grid..."
    )

    merged = xr.Dataset(
        {
            "sst": sst,
            "salinity": salinity,
            "u_current": u_current,
            "v_current": v_current,
            "swh": swh,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "pressure": pressure,
            "bathymetry": bathymetry,
            "land_mask": land_mask,
        }
    )

    print(
        "Merged dataset dims:",
        dict(merged.sizes)
    )

    # =========================================================
    # Restrict to the common temporal overlap
    # =========================================================
    # =========================================================
# Restrict to the final common project period
# =========================================================
    common_start = "2022-11-01"
    common_end = "2024-12-31"

    print(
        f"Restricting to common project period: "
        f"{common_start} -> {common_end}"
    )

    merged = merged.sel(
        time=slice(common_start, common_end)
    )

    print(
        "After common-time restriction:",
        dict(merged.sizes)
    )

    print(
        "After common-time restriction:",
        dict(merged.sizes)
    )

    print(
        "  >> Compare this against expectations "
        "from Step 0/4 before proceeding."
    )

    # =========================================================
    # 5. Fill short transient gaps
    # =========================================================
    print(
        "Gap-filling short transient NaNs "
        "(maximum 3 daily values)..."
    )

    vars_to_fill = [
        "sst",
        "salinity",
        "u_current",
        "v_current",
        "swh",
        "wind_speed",
        "wind_direction",
        "pressure",
    ]

    for var in vars_to_fill:

        merged[var] = fill_short_gaps(
            merged[var],
            max_gap_days=3,
        )

    # =========================================================
    # 6. Check remaining SST NaNs at ocean points
    # =========================================================
    ocean_sst = merged["sst"].where(
        merged["land_mask"] == 1
    )

    remaining_nans = int(
        ocean_sst.isnull().sum()
    )

    print(
        "Remaining NaNs at ocean points "
        f"after gap-filling (sst): {remaining_nans}"
    )

    if remaining_nans > 0:
        print(
            "  >> Investigate these dates/locations "
            "before proceeding if this number looks large."
        )

    # =========================================================
    # 7. Save final merged dataset
    # =========================================================
    MERGED_NC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_netcdf(
        MERGED_NC_PATH
    )

    print(
        f"\nSaved merged dataset to: "
        f"{MERGED_NC_PATH}"
    )


if __name__ == "__main__":
    main()