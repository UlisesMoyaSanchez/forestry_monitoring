#!/usr/bin/env python3
"""Build the curated Satellite Image Time Series (SITS) dataset for the CCAI
tutorial "Deforestation detection from satellite image time series".

CREATOR-SIDE SCRIPT. Tutorial participants do NOT run this: they download the
small CSV it produces. We keep it in the repo so the dataset is fully
reproducible and its provenance is transparent (a requirement of the CCAI
"real-world data conditions" and datasheet guidelines).

What it does
------------
1. Downloads real deforestation labels (polygons) from INPE's PRODES program
   via the open TerraBrasilis WFS service.
2. Loads a small Sentinel-2 L2A data cube for one year over an active
   deforestation frontier via the Microsoft Planetary Computer STAC API.
3. Masks clouds (Scene Classification Layer), computes NDVI, and resamples each
   pixel to a regular temporal grid with gap-filling.
4. Samples labelled pixel time series (stable forest vs. deforested-this-year,
   optionally an older-clearing / pasture class) and writes a tidy CSV plus a
   metadata.json describing exactly how it was made.

Run:
    python data_prep/build_dataset.py --out data/amazon_sits_samples.csv

Requires the `forestry-ccai` conda env (see environment.yml).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (defaults chosen for a fast, reproducible teaching dataset)
# ---------------------------------------------------------------------------

# A small window on an active deforestation frontier in Para (PA), Brazil, near
# the Rio Novo / BR-163 arc. Kept small so the cube stays a few hundred MB and
# the build is quick. Verified to contain multi-year PRODES polygons + forest.
DEFAULT_BBOX = (-54.75, -3.70, -54.55, -3.50)  # (minlon, minlat, maxlon, maxlat)

# PRODES "year" runs on a ~Aug->Jul calendar. To *see* a clearing happen inside
# the time series, we observe the 12 months centred on the clearing detection.
TARGET_YEAR = 2022
SEASON_START = f"{TARGET_YEAR - 1}-08-01"
SEASON_END = f"{TARGET_YEAR}-07-31"

# Regular temporal grid the messy Sentinel-2 revisit is resampled onto.
GRID_DAYS = 15  # ~24 steps across the year

# NOTE: the layer's native CRS is EPSG:4674 (lat/lon axis order). The plain
# `bbox=` KVP does not filter reliably, but a CQL BBOX() spatial filter does --
# and it expects the box in lat,lon,lat,lon order.
PRODES_WFS = (
    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-amazon-nb/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=prodes-amazon-nb:{layer}"
    "&outputFormat=application/json&srsName=EPSG:4326"
    "&count={count}&CQL_FILTER=BBOX(geom,{miny},{minx},{maxy},{maxx})"
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Sentinel-2 SCL classes we treat as valid (not cloud/shadow/snow/nodata).
# 4=vegetation, 5=bare soil, 6=water. Class 7 (unclassified) is deliberately
# EXCLUDED: it is dominated by cloud edges and haze, which systematically
# depress NDVI through the Amazon wet season.
SCL_VALID = {4, 5, 6}


# ---------------------------------------------------------------------------
# Step 1 - labels from PRODES (TerraBrasilis WFS)
# ---------------------------------------------------------------------------
def fetch_prodes(bbox, count=4000):
    """Return (yearly_gdf, accumulated_gdf) clipped to bbox, in EPSG:4326."""
    import geopandas as gpd

    minx, miny, maxx, maxy = bbox
    frames = {}
    for key, layer in [
        ("yearly", "yearly_deforestation_biome"),
        ("accum", "accumulated_deforestation_2007_biome"),
    ]:
        url = PRODES_WFS.format(
            layer=layer, count=count,
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        )
        print(f"  [PRODES] fetching {layer} ...", flush=True)
        gdf = gpd.read_file(url)
        if len(gdf):
            gdf = gdf.set_crs(4326, allow_override=True)
        frames[key] = gdf
        print(f"           {len(gdf)} polygons", flush=True)
    return frames["yearly"], frames["accum"]


# ---------------------------------------------------------------------------
# Step 2 - Sentinel-2 data cube via Planetary Computer STAC
# ---------------------------------------------------------------------------
def load_cube(bbox, start, end, resolution=20):
    """Load a lazy NDVI + SCL cube (time, y, x) for the bbox and season."""
    import planetary_computer as pc
    import pystac_client
    from odc.stac import stac_load

    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 80}},
    )
    items = list(search.items())
    print(f"  [STAC] {len(items)} Sentinel-2 scenes found", flush=True)
    if not items:
        raise RuntimeError("No Sentinel-2 scenes for this bbox/season.")

    # stac_load takes `resolution` in target-CRS units; ours is EPSG:4326, so
    # convert the metre pixel size to degrees (~111.32 km per degree).
    res_deg = resolution / 111_320.0
    cube = stac_load(
        items,
        bands=["red", "nir", "SCL"],
        bbox=bbox,
        resolution=res_deg,
        crs="EPSG:4326",
        chunks={},  # dask-lazy
        groupby="solar_day",
    )
    return cube


def cube_to_ndvi(cube):
    """Cloud-mask via SCL, compute NDVI, resample to a regular temporal grid."""
    scl = cube["SCL"]
    valid = scl.isin(list(SCL_VALID))

    red = cube["red"].where(valid)
    nir = cube["nir"].where(valid)
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.clip(-1, 1).rename("ndvi")

    # Resample messy revisit onto a regular grid using a maximum-value
    # composite (MVC): the max NDVI in each window comes from the clearest
    # observation, so residual haze/cloud that slipped past the SCL mask does
    # not drag the composite down. Then load into memory before gap-filling
    # (interpolate_na needs a single chunk along the time axis).
    ndvi = ndvi.resample(time=f"{GRID_DAYS}D").max()
    print("  [NDVI] loading cube into memory ...", flush=True)
    ndvi = ndvi.compute()
    ndvi = ndvi.interpolate_na(dim="time", method="linear").ffill("time").bfill("time")
    return ndvi


# ---------------------------------------------------------------------------
# Step 3 - sample labelled pixel time series
# ---------------------------------------------------------------------------
def sample_points(gdf, n, rng, min_area_km=0.05):
    """Sample representative interior points from polygons (area-weighted)."""
    if gdf is None or len(gdf) == 0:
        return []
    g = gdf.copy()
    if "area_km" in g.columns:
        g = g[g["area_km"] >= min_area_km]
    if len(g) == 0:
        return []
    # Buffer inward a little (~30 m ~ 0.0003 deg) to avoid mixed edge pixels.
    # buffer(0) first repairs any invalid polygons.
    interior = g.geometry.buffer(0).buffer(-0.0003)
    interior = interior[~interior.is_empty]
    pts = []
    idx = rng.choice(interior.index.values, size=min(n, len(interior)), replace=len(interior) < n)
    for i in idx:
        geom = interior.loc[i]
        if hasattr(geom, "iloc"):
            geom = geom.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        for _ in range(50):
            from shapely.geometry import Point
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if geom.contains(p):
                pts.append((p.x, p.y))
                break
    return pts


def extract_series(ndvi, points, label, rng, jitter=True):
    """Extract the NDVI time series nearest each (lon, lat) point."""
    rows = []
    for lon, lat in points:
        series = ndvi.sel(longitude=lon, latitude=lat, method="nearest").values
        if np.isnan(series).all():
            continue
        rows.append((lon, lat, label, series.astype(np.float32)))
    return rows


def build_forest_points(bbox, accum_gdf, n, rng):
    """Random points inside bbox that fall OUTSIDE all known deforestation."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    minx, miny, maxx, maxy = bbox
    # buffer(0) repairs invalid self-intersecting polygons before the union.
    cleared = None
    if len(accum_gdf):
        cleared = unary_union([g.buffer(0) for g in accum_gdf.geometry.values])
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 200:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if cleared is None or not cleared.buffer(0.0003).contains(p):
            pts.append((p.x, p.y))
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/amazon_sits_samples.csv")
    ap.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX))
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--resolution", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bbox = tuple(args.bbox)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("1) Downloading PRODES labels ...", flush=True)
    yearly, accum = fetch_prodes(bbox)

    print("2) Loading Sentinel-2 cube ...", flush=True)
    cube = load_cube(bbox, SEASON_START, SEASON_END, resolution=args.resolution)
    ndvi = cube_to_ndvi(cube)
    times = [np.datetime_as_string(t, unit="D") for t in ndvi["time"].values]
    print(f"   temporal grid: {len(times)} steps ({times[0]} .. {times[-1]})", flush=True)

    print("3) Sampling labelled time series ...", flush=True)
    rows = []

    # deforested THIS year
    defo = yearly[yearly["year"] == TARGET_YEAR] if "year" in yearly.columns else yearly
    rows += extract_series(ndvi, sample_points(defo, args.per_class, rng), "deforested", rng)

    # cleared years ago (older PRODES years) -> pasture-like trajectory
    old = yearly[yearly["year"] <= TARGET_YEAR - 4] if "year" in yearly.columns else yearly.iloc[0:0]
    rows += extract_series(ndvi, sample_points(old, args.per_class, rng), "old_clearing", rng)

    # stable forest
    rows += extract_series(ndvi, build_forest_points(bbox, accum, args.per_class, rng), "forest", rng)

    if not rows:
        sys.exit("No samples extracted - check bbox / connectivity.")

    n_t = len(times)
    cols = ["lon", "lat", "label"] + [f"ndvi_{i:02d}" for i in range(n_t)]
    data = []
    for lon, lat, label, series in rows:
        series = np.asarray(series, dtype=np.float32)
        if len(series) != n_t:
            continue
        data.append([lon, lat, label, *series.tolist()])
    df = pd.DataFrame(data, columns=cols).dropna()

    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} samples -> {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(df["label"].value_counts().to_string())

    meta = {
        "created": datetime.utcnow().isoformat() + "Z",
        "bbox_lonlat": bbox,
        "target_prodes_year": TARGET_YEAR,
        "season": [SEASON_START, SEASON_END],
        "temporal_grid_days": GRID_DAYS,
        "time_steps": times,
        "resolution_m": args.resolution,
        "labels": sorted(df["label"].unique().tolist()),
        "n_samples": len(df),
        "sources": {
            "imagery": "Sentinel-2 L2A via Microsoft Planetary Computer STAC",
            "labels": "INPE PRODES yearly_deforestation_biome (TerraBrasilis WFS)",
        },
    }
    (out.parent / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote provenance -> {out.parent / 'metadata.json'}")


if __name__ == "__main__":
    main()
