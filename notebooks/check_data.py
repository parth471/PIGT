import xarray as xr

file_path = "data/raw/arabian_sea_physics.nc"

ds = xr.open_dataset(file_path)

print(ds)
print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))