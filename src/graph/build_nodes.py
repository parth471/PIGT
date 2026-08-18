import xarray as xr
import numpy as np
import pandas as pd

# Load dataset
ds = xr.open_dataset("data/raw/arabian_sea_physics.nc")

# Surface data
thetao = ds["thetao"].isel(depth=0)

# Use first timestep to identify valid ocean locations
mask = np.isfinite(thetao.isel(time=0).values)

latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

# Get grid indices of valid ocean points
lat_idx, lon_idx = np.where(mask)

# Create node table
nodes = pd.DataFrame({
    "node_id": np.arange(len(lat_idx)),
    "lat_index": lat_idx,
    "lon_index": lon_idx,
    "latitude": latitudes[lat_idx],
    "longitude": longitudes[lon_idx]
})

print("Number of nodes:", len(nodes))
print("\nFirst 10 nodes:")
print(nodes.head(10))

# Save
nodes.to_csv("data/processed_nodes.csv", index=False)

print("\nSaved to: data/processed_nodes.csv")