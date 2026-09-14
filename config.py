"""
Central configuration for the PIGT Bay of Bengal data pipeline (Person 1).
Edit values here — every other script imports from this file, so you only
need to change settings in one place.
"""
from pathlib import Path

# ============================================================
# REGION  (Bay of Bengal)
# ============================================================
LAT_MIN, LAT_MAX = 10, 20
LON_MIN, LON_MAX = 80, 90

# ============================================================
# TIME RANGE  (first working version: 3 years)
# ============================================================
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# ============================================================
# TEMPORAL SEQUENCE CONFIG
# ============================================================
HISTORY_DAYS = 7             # past N days used as input
FORECAST_HORIZON_DAYS = 1    # predict 1 day ahead

# ============================================================
# CHRONOLOGICAL SPLIT FRACTIONS
# ============================================================
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remaining 0.15

# ============================================================
# FEATURE / TARGET ORDER  (fixed — do not reorder without updating
# every script and telling Person 2)
# ============================================================
# The "previous ocean states" requirement is satisfied by including
# the 5 target variables as part of every day's feature vector across
# the whole history window — i.e. the model literally sees the past
# N days of SST/salinity/currents/SWH as input, then predicts day N+1.
FEATURE_ORDER = [
    "sst", "salinity", "u_current", "v_current", "swh",   # previous ocean states
    "wind_speed", "wind_direction", "pressure",           # atmospheric forcing
    "bathymetry", "latitude", "longitude",                # static / spatial
    "sin_day_of_year", "cos_day_of_year",                 # seasonal encoding
]
TARGET_ORDER = ["sst", "salinity", "u_current", "v_current", "swh"]

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent   # .../PIGT/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"

RAW_PHYSICS_DIR = RAW_DIR / "copernicus_physics"
RAW_WAVE_DIR = RAW_DIR / "copernicus_wave"
RAW_ERA5_DIR = RAW_DIR / "era5"

MERGED_NC_PATH = PROCESSED_DIR / "merged_bob_daily.nc"

# Create all directories up front so scripts never fail on a missing folder
for d in [RAW_PHYSICS_DIR, RAW_WAVE_DIR, RAW_ERA5_DIR, PROCESSED_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
