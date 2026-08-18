import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load data
ds = xr.open_dataset("data/raw/arabian_sea_physics.nc")

# Use temperature from the first day
thetao = ds["thetao"].isel(time=0, depth=0)

# Valid ocean points = points where temperature exists
ocean_mask = np.isfinite(thetao.values)

print("Grid shape:", ocean_mask.shape)
print("Ocean points:", ocean_mask.sum())
print("Land/missing points:", (~ocean_mask).sum())

# Visualize the mask
plt.figure(figsize=(10, 6))
plt.imshow(
    ocean_mask,
    origin="lower",
    extent=[
        float(ds.longitude.min()),
        float(ds.longitude.max()),
        float(ds.latitude.min()),
        float(ds.latitude.max())
    ],
    aspect="auto"
)

plt.title("Ocean Grid Mask")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()