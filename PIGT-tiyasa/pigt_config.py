"""Project configuration for the PIGT ocean-data preparation pipeline."""
from pathlib import Path

# -----------------------------------------------------------------------------
# Project folders
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PHYSICS_DIR = RAW_DIR / "copernicus_physics"
RAW_WAVE_DIR = RAW_DIR / "copernicus_wave"
RAW_ERA5_DIR = RAW_DIR / "era5"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
GRAPH_DIR = DATA_DIR / "graph"
REPORT_DIR = ROOT / "results" / "data_pipeline"

# -----------------------------------------------------------------------------
# Raw files actually present in the project
# -----------------------------------------------------------------------------
RAW_FILES = {
    "sst": RAW_PHYSICS_DIR / "thetao_bob.nc",
    "salinity": RAW_PHYSICS_DIR / "so_bob.nc",
    "currents": RAW_PHYSICS_DIR / "currents_bob.nc",
    "bathymetry": RAW_PHYSICS_DIR / "bathymetry.nc",
    "wave": RAW_WAVE_DIR / "vhm0_bob.nc",
    "era5": RAW_ERA5_DIR / "era5_bob.nc",
}

# -----------------------------------------------------------------------------
# Region
# -----------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 10.0, 20.0
LON_MIN, LON_MAX = 80.0, 90.0

# Do not hard-code the common period in the processing logic. The pipeline
# calculates it from the actual raw files and then clips it to these optional
# project bounds.
PROJECT_START = "2022-01-01"
PROJECT_END = "2024-12-31"

# -----------------------------------------------------------------------------
# Temporal protocol
# -----------------------------------------------------------------------------
FREQUENCY = "1D"
HISTORY_DAYS = 7                 # t-6 ... t
FORECAST_HORIZON_DAYS = 1        # predict t+1
MAX_GAP_DAYS = 3                 # existing short-gap cleaning rule

# -----------------------------------------------------------------------------
# Chronological split
# -----------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# -----------------------------------------------------------------------------
# Exact model contract
# -----------------------------------------------------------------------------
FEATURE_ORDER = [
    "sst",
    "salinity",
    "u_current",
    "v_current",
    "swh",
    "wind_speed",
    "wind_direction",
    "pressure",
    "bathymetry",
    "latitude",
    "longitude",
    "sin_day_of_year",
    "cos_day_of_year",
]

TARGET_ORDER = [
    "sst",
    "salinity",
    "u_current",
    "v_current",
    "swh",
    "wind_speed",
    "wind_direction",
    "pressure",
]

assert len(FEATURE_ORDER) == 13
assert len(TARGET_ORDER) == 8

# -----------------------------------------------------------------------------
# Spatial graph
# -----------------------------------------------------------------------------
K_NEIGHBORS = 8

# -----------------------------------------------------------------------------
# Audit ranges. These are ONLY used to flag suspicious ocean values; they are
# not used to clip or alter the production data automatically.
# -----------------------------------------------------------------------------
VALID_RANGES = {
    "sst": (-5.0, 45.0),
    "salinity": (0.0, 50.0),
    "u_current": (-5.0, 5.0),
    "v_current": (-5.0, 5.0),
    "swh": (0.0, 30.0),
    "wind_speed": (0.0, 100.0),
    "wind_direction": (0.0, 360.0),
    "pressure": (800.0, 1100.0),
    "bathymetry": (0.0, 12000.0),
}

# Values below this standard deviation are treated as insufficient useful
# variation for the augmentation decision. This is deliberately conservative.
MIN_USEFUL_STD = 1e-6

# Create output directories when the configuration is imported.
for _directory in (PROCESSED_DIR, FINAL_DIR, GRAPH_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
