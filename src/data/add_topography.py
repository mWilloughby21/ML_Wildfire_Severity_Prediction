# add_topography.py
#
# ============================================================
# NEW FILE: pulls elevation per fire from OpenTopoData's free
# public SRTM 30m endpoint, then derives slope and aspect from
# a 3x3 kernel sampled around each fire location. Adds four new
# columns to data/processed/cleaned_data.csv:
#   ELEVATION, SLOPE, ASPECT_SIN, ASPECT_COS
# Aspect is encoded as sin/cos because it's a circular variable.
# ============================================================

# python -m src.data.add_topography

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
API_URL = "https://api.opentopodata.org/v1/srtm30m"

# Approx. 100 m kernel spacing for slope/aspect estimation.
# Using meters keeps slope in real degrees regardless of latitude.
KERNEL_SPACING_M = 100
BATCH_SIZE = 100          # OpenTopoData allows up to 100 locations/call
RATE_LIMIT_SECONDS = 1.0  # public endpoint: 1 call/sec


def fetch_elevations(coords):
    """coords is a list of (lat, lon) tuples; returns list of elevations (m)."""
    elevations = []
    for batch_start in range(0, len(coords), BATCH_SIZE):
        batch = coords[batch_start:batch_start + BATCH_SIZE]
        loc = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in batch)
        url = f"{API_URL}?{urllib.parse.urlencode({'locations': loc})}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        if data.get('status') != 'OK':
            raise RuntimeError(f"OpenTopoData error: {data}")
        # OpenTopoData returns null for points outside the dataset; coerce to NaN.
        elevations.extend(
            (float(r['elevation']) if r['elevation'] is not None else float('nan'))
            for r in data['results']
        )
        done = batch_start + len(batch)
        print(f"  fetched {done}/{len(coords)} elevations")
        if done < len(coords):
            time.sleep(RATE_LIMIT_SECONDS)
    return elevations


def kernel_offsets(lat_deg):
    """Return (dlat, dlon) in degrees for a ~100 m kernel step at this latitude."""
    dlat = KERNEL_SPACING_M / 111_000.0
    dlon = KERNEL_SPACING_M / (111_000.0 * math.cos(math.radians(lat_deg)))
    return dlat, dlon


# Horn's method: standard GIS slope/aspect from a 3x3 elevation kernel.
def slope_aspect_from_kernel(elev_3x3, cell_size_m):
    a, b, c = elev_3x3[0]
    d, _, f = elev_3x3[1]
    g, h, i = elev_3x3[2]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_size_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell_size_m)
    slope_deg = math.degrees(math.atan(math.sqrt(dzdx ** 2 + dzdy ** 2)))
    aspect_deg = (math.degrees(math.atan2(dzdy, -dzdx)) + 360.0) % 360.0
    return slope_deg, aspect_deg


def add_topography():
    df = pd.read_csv(CLEANED_DATA_PATH)
    n = len(df)
    print(f"Building 3x3 query grid for {n} fires ({n * 9} elevation points)...")

    coords = []
    for _, fire in df.iterrows():
        lat, lon = fire['LATITUDE'], fire['LONGITUDE']
        dlat, dlon = kernel_offsets(lat)
        for ki in (-1, 0, 1):       # row: north -> south
            for kj in (-1, 0, 1):   # col: west -> east
                coords.append((lat - ki * dlat, lon + kj * dlon))

    print(f"Fetching elevations from OpenTopoData ({len(coords)} points)...")
    elevations = fetch_elevations(coords)

    print("Computing slope and aspect...")
    elevs, slopes, aspects = [], [], []
    for i in range(n):
        kernel = np.array(elevations[i * 9:(i + 1) * 9]).reshape(3, 3)
        center = kernel[1, 1]
        if np.isnan(kernel).any():
            slope, aspect = float('nan'), float('nan')
        else:
            slope, aspect = slope_aspect_from_kernel(kernel.tolist(), KERNEL_SPACING_M)
        elevs.append(center)
        slopes.append(slope)
        aspects.append(aspect)

    df['ELEVATION'] = elevs
    df['SLOPE'] = slopes
    aspect_rad = np.radians(np.array(aspects))
    df['ASPECT_SIN'] = np.sin(aspect_rad)   # cyclical encoding (north vs south facing)
    df['ASPECT_COS'] = np.cos(aspect_rad)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Wrote {CLEANED_DATA_PATH} with {df.shape[1]} columns "
          f"(added ELEVATION, SLOPE, ASPECT_SIN, ASPECT_COS).")
    print(df[['ELEVATION', 'SLOPE', 'ASPECT_SIN', 'ASPECT_COS']].describe())


if __name__ == "__main__":
    add_topography()
