"""
STEP 1b — Download ERA5 wind + pressure data for the Bay of Bengal region.

PREREQUISITES:
1. pip install cdsapi
2. Register at https://cds.climate.copernicus.eu
3. Get your Personal Access Token from https://cds.climate.copernicus.eu/profile
4. Create a file at ~/.cdsapirc (Windows: C:\\Users\\<you>\\.cdsapirc) containing:

       url: https://cds.climate.copernicus.eu/api
       key: <PASTE-YOUR-PERSONAL-ACCESS-TOKEN-HERE>

5. Open the "ERA5 hourly data on single levels from 1940 to present" dataset
   page on the CDS website and accept its Terms of Use once (manual, one-time).

USAGE:
    python download_era5.py

NOTE: This queues an asynchronous request on ECMWF's servers. It can take
anywhere from a couple of minutes to an hour depending on server load — the
script will wait and download automatically once it's ready.
"""
import cdsapi

from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    START_DATE, END_DATE, RAW_ERA5_DIR,
)

start_year = int(START_DATE[:4])
end_year = int(END_DATE[:4])

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ],
    "year": [str(y) for y in range(start_year, end_year + 1)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": ["00:00", "06:00", "12:00", "18:00"],  # 6-hourly; enough for daily means
    "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # [North, West, South, East]
    "data_format": "netcdf",
}

output_path = RAW_ERA5_DIR / "era5_bob.nc"

if __name__ == "__main__":
    print(f"Requesting ERA5 data for {start_year}-{end_year} over "
          f"lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}]...")
    print("This may take a while — the request is queued on ECMWF's servers.")
    client.retrieve(dataset, request, str(output_path))
    print(f"\nDone. Saved to: {output_path}")
