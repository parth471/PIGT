"""
STEP 4 — Inspect every downloaded raw file BEFORE doing anything else with it.

USAGE:
    python 01_inspect_raw_data.py
"""
import xarray as xr

from config import RAW_PHYSICS_DIR, RAW_WAVE_DIR, RAW_ERA5_DIR


def inspect(path):
    print("=" * 70)
    print(path)
    print("=" * 70)
    try:
        ds = xr.open_dataset(path)
    except FileNotFoundError:
        print("  !! FILE NOT FOUND — did the download step complete? Skipping.\n")
        return

    print("\n--- Dimensions ---")
    print(dict(ds.dims))

    print("\n--- Coordinates ---")
    print(list(ds.coords))

    print("\n--- Data variables ---")
    print(list(ds.data_vars))

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    print(f"\n--- {lat_name}/{lon_name} range ---")
    print(f"{lat_name}: {float(ds[lat_name].min())} to {float(ds[lat_name].max())}")
    print(f"{lon_name}: {float(ds[lon_name].min())} to {float(ds[lon_name].max())}")

    time_name = "time" if "time" in ds.coords else "valid_time"
    if time_name in ds.coords:
        print(f"\n--- Time range ({time_name}) ---")
        print("start:", ds[time_name].values.min())
        print("end:  ", ds[time_name].values.max())
        print("n timesteps:", ds[time_name].size)
    else:
        print("\n--- No time coordinate found (check coordinate names!) ---")

    print("\n--- Variable units ---")
    for v in ds.data_vars:
        print(f"  {v} -> {ds[v].attrs.get('units', 'NO UNITS FOUND')}")

    print("\n--- NaN check (first data variable) ---")
    first_var = list(ds.data_vars)[0]
    nan_frac = float(ds[first_var].isnull().mean())
    print(f"  {first_var}: {nan_frac:.2%} NaN "
          f"(expected > 0% if it's an ocean variable — that's land)")

    if "depth" in ds.coords:
        print("\n--- Depth levels ---")
        depth_vals = ds.depth.values
        print(depth_vals[:10], "..." if len(depth_vals) > 10 else "")
        if len(depth_vals) > 1:
            print("  >> WARNING: more than 1 depth level present. "
                  "Re-check --minimum-depth/--maximum-depth in the download step.")

    ds.close()
    print()


if __name__ == "__main__":
    files = [
        RAW_PHYSICS_DIR / "thetao_bob.nc",
        RAW_PHYSICS_DIR / "so_bob.nc",
        RAW_PHYSICS_DIR / "currents_bob.nc",
        RAW_PHYSICS_DIR / "bathymetry_bob.nc",
        RAW_WAVE_DIR / "vhm0_bob.nc",
        RAW_ERA5_DIR / "era5_bob.nc",
    ]
    for f in files:
        inspect(f)

    print("Inspection complete. Review the printed ranges/units/NaN fractions "
          "against Step 0 expectations before moving to the next script.")
