"""
STEP 1a — Download Copernicus Marine Physics + Wave + static Bathymetry
for the Bay of Bengal region.

PREREQUISITES:
1. pip install copernicusmarine
2. Register a free account at https://data.marine.copernicus.eu
3. Run once in your terminal (stores credentials so you never re-enter them):
       copernicusmarine login

USAGE:
    python download_copernicus_marine.py
"""
import copernicusmarine

from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    START_DATE, END_DATE,
    RAW_PHYSICS_DIR, RAW_WAVE_DIR,
)

START_DATETIME = f"{START_DATE}T00:00:00"
END_DATETIME = f"{END_DATE}T23:59:59"


def download_thetao():
    print("\n[1/5] Downloading SST (thetao)...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=START_DATETIME,
        end_datetime=END_DATETIME,
        output_directory=str(RAW_PHYSICS_DIR),
        output_filename="thetao_bob.nc",
        force_download=True,
    )


def download_so():
    print("\n[2/5] Downloading Salinity (so)...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=["so"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1,
        start_datetime=START_DATETIME, end_datetime=END_DATETIME,
        output_directory=str(RAW_PHYSICS_DIR),
        output_filename="so_bob.nc",
        force_download=True,
    )


def download_currents():
    print("\n[3/5] Downloading Currents (uo, vo)...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1,
        start_datetime=START_DATETIME, end_datetime=END_DATETIME,
        output_directory=str(RAW_PHYSICS_DIR),
        output_filename="currents_bob.nc",
        force_download=True,
    )


def download_bathymetry():
    print("\n[4/5] Downloading static bathymetry + land/sea mask...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
        dataset_part="bathy",
        variables=["deptho"],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        output_directory=str(RAW_PHYSICS_DIR),
        output_filename="bathymetry.nc",
        force_download=True,
    )


def download_wave():
    print("\n[5/5] Downloading Significant Wave Height (VHM0)...")
    print(
        "NOTE: if this fails with 'dataset not found', the wave sub-dataset ID may have\n"
        "changed. Verify it first by running in a terminal:\n"
        "    copernicusmarine describe --contains GLOBAL_ANALYSISFORECAST_WAV_001_027\n"
        "and update WAVE_DATASET_ID below to match."
    )
    WAVE_DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"  # verify before running!
    copernicusmarine.subset(
        dataset_id=WAVE_DATASET_ID,
        variables=["VHM0"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=START_DATETIME, end_datetime=END_DATETIME,
        output_directory=str(RAW_WAVE_DIR),
        output_filename="vhm0_bob.nc",
        force_download=True,
    )


if __name__ == "__main__":
    download_thetao()
    download_so()
    download_currents()
    download_bathymetry()
    download_wave()
    print("\nAll Copernicus Marine downloads complete.")
    print(f"Physics files in: {RAW_PHYSICS_DIR}")
    print(f"Wave file in:     {RAW_WAVE_DIR}")
