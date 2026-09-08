"""
fetch_real_data.py — run this ONCE you've done `copernicusmarine login`
====================================================================
This CMEMS product splits variables across SEPARATE dataset IDs rather
than bundling them into one -- thetao, so, and currents (uo/vo) each
live in their own 3D dataset, while zos (2D, surface only) lives in a
different one with no variable suffix. So this script does 4 downloads,
then merges them into one file: data/raw/cmems_subset.nc

Run it directly:
    python3 src/graph/fetch_real_data.py

After it finishes, open build_nodes.py and change:
    USE_SYNTHETIC = True   ->   USE_SYNTHETIC = False
then re-run the full pipeline from build_nodes.py onward.
"""

import copernicusmarine
import xarray as xr
from pathlib import Path

LAT_MIN, LAT_MAX = 10.0, 20.0
LON_MIN, LON_MAX = 80.0, 90.0
START, END = "2024-01-01T00:00:00", "2024-06-30T00:00:00"
RAW_DIR = "data/raw"

# each entry: (dataset_id, variables, has_depth_dimension)
DOWNLOADS = [
    ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"], True),
    ("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ["so"], True),
    ("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"], True),
    ("cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ["zos"], False),
]


def main():
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for dataset_id, variables, has_depth in DOWNLOADS:
        out_name = f"{'_'.join(variables)}.nc"
        print(f"[INFO] Downloading {variables} from {dataset_id} ...")
        kwargs = dict(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
            start_datetime=START, end_datetime=END,
            output_filename=out_name,
            output_directory=RAW_DIR,
        )
        if has_depth:
            kwargs.update(minimum_depth=0, maximum_depth=1)  # surface layer only

        copernicusmarine.subset(**kwargs)
        downloaded_files.append(f"{RAW_DIR}/{out_name}")

    print("[INFO] Merging the 4 downloads into one file ...")
    datasets = [xr.open_dataset(f) for f in downloaded_files]
    # thetao/so/cur have a depth dimension (single surface layer) -- drop it so
    # everything lines up on (time, latitude, longitude) before merging with zos
    datasets = [ds.squeeze("depth", drop=True) if "depth" in ds.dims else ds for ds in datasets]
    merged = xr.merge(datasets)
    merged.to_netcdf(f"{RAW_DIR}/cmems_subset.nc")

    print(f"[INFO] Saved merged real CMEMS subset to {RAW_DIR}/cmems_subset.nc")
    print(f"[INFO] Variables present: {list(merged.data_vars)}")
    print("[INFO] Now edit build_nodes.py: set USE_SYNTHETIC = False, RAW_NC_PATH = 'data/raw/cmems_subset.nc', then re-run the pipeline.")


if __name__ == "__main__":
    main()