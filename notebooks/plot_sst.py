import xarray as xr
import matplotlib.pyplot as plt

# Load dataset
ds = xr.open_dataset("data/raw/arabian_sea_physics.nc")

# Get surface temperature for the first day
sst = ds["thetao"].isel(time=0, depth=0)

# Plot
plt.figure(figsize=(10, 6))

sst.plot()

plt.title("Arabian Sea Surface Temperature - 1 January 2025")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()