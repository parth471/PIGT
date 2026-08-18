import pandas as pd
import numpy as np
import xarray as xr

# -----------------------------
# Load data
# -----------------------------
ds = xr.open_dataset("data/raw/arabian_sea_physics.nc")

latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

# Load node table
nodes = pd.read_csv("data/processed_nodes.csv")

# Map (lat_index, lon_index) -> node_id
node_lookup = {
    (row.lat_index, row.lon_index): int(row.node_id)
    for row in nodes.itertuples()
}

# -----------------------------
# Haversine distance
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate geographic distance between two coordinates.
    Returns distance in kilometers.
    """

    R = 6371.0

    # Convert coordinates to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))

# -----------------------------
# 8-neighbor directions
# -----------------------------
neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

edges = []

# -----------------------------
# Build edges
# -----------------------------
for row in nodes.itertuples():

    node_id = int(row.node_id)
    i = int(row.lat_index)
    j = int(row.lon_index)

    lat1 = float(row.latitude)
    lon1 = float(row.longitude)

    for di, dj in neighbors:

        neighbor_position = (i + di, j + dj)

        # Skip if neighboring grid cell is land/missing
        if neighbor_position not in node_lookup:
            continue

        neighbor_id = node_lookup[neighbor_position]

        lat2 = float(latitudes[i + di])
        lon2 = float(longitudes[j + dj])

        distance_km = haversine(
            lat1, lon1,
            lat2, lon2
        )

        edges.append({
            "source": node_id,
            "target": neighbor_id,
            "distance_km": distance_km
        })


# -----------------------------
# Save edges
# -----------------------------
edges_df = pd.DataFrame(edges)

edges_df.to_csv(
    "data/edges.csv",
    index=False
)

print("Number of directed edges:", len(edges_df))

print("\nFirst 10 edges:")
print(edges_df.head(10))

print("\nDistance statistics:")
print(edges_df["distance_km"].describe())

print("\nSaved to: data/edges.csv")