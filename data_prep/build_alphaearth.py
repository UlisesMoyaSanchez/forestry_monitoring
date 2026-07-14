#!/usr/bin/env python3
"""Build the AlphaEarth (Google Satellite Embedding) companion dataset for the
CCAI tutorial "Deforestation detection from satellite image time series".

CREATOR-SIDE SCRIPT. Tutorial participants do NOT run this: they download the
small CSV it produces. Unlike build_dataset.py (which needs no account), this
script requires Google Earth Engine access, which is why the embeddings are
pre-extracted and shipped as a CSV -- participants stay auth-free.

What it does
------------
1. Reads the sample points (lon, lat, label) from the existing NDVI dataset
   (data/amazon_sits_samples.csv) so both CSVs are row-aligned.
2. For each year, samples the 64-band annual embedding image from
   GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (AlphaEarth Foundations; Brown et
   al. 2025) at those points, scale=10 m.
3. Writes a wide CSV (lon, lat, label, emb_<year>_00 .. emb_<year>_63 for each
   year) plus alphaearth_metadata.json describing exactly how it was made.

Years 2018-2024 are extracted on purpose: the PRODES target year is 2022, so
2023-2024 embeddings "see" the already-cleared land. The notebook defaults to
2018-2022 and uses the later years only for an explicit label-leakage exercise.

Setup (one-time, creator only):
    pip install earthengine-api
    earthengine authenticate

Run:
    python data_prep/build_alphaearth.py --project <your-gcloud-project>
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
# Configuration
# ---------------------------------------------------------------------------

EE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
N_DIMS = 64  # bands A00 .. A63, each in [-1, 1]
SCALE_M = 10

# PRODES target year of the NDVI dataset is 2022. We extract a symmetric-ish
# window around it; see the leakage note in the module docstring.
DEFAULT_YEARS = list(range(2018, 2025))

# sampleRegions can drop points (masked pixels) and does not guarantee output
# order, so every point carries its source row index and we re-join on it.
IDX_PROP = "idx"


def build_feature_collection(df):
    import ee

    feats = [
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {IDX_PROP: int(i)})
        for i, r in df.iterrows()
    ]
    return ee.FeatureCollection(feats)


def sample_year(fc, year, n_points):
    """Return an (n_points, 64) array of embeddings for one year (NaN = masked)."""
    import ee

    band_names = [f"A{b:02d}" for b in range(N_DIMS)]
    img = (
        ee.ImageCollection(EE_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
    )
    sampled = img.sampleRegions(collection=fc, scale=SCALE_M, geometries=False)
    feats = sampled.getInfo()["features"]

    mat = np.full((n_points, N_DIMS), np.nan, dtype=np.float32)
    for f in feats:
        props = f["properties"]
        mat[int(props[IDX_PROP])] = [props[b] for b in band_names]
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/amazon_sits_samples.csv",
                    help="existing NDVI sample CSV providing lon/lat/label")
    ap.add_argument("--out", default="data/amazon_alphaearth_samples.csv")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--project", default=None,
                    help="Google Cloud project for ee.Initialize()")
    args = ap.parse_args()

    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api is not installed. Run: pip install earthengine-api")

    ee.Initialize(project=args.project)

    df = pd.read_csv(args.samples)[["lon", "lat", "label"]]
    print(f"Loaded {len(df)} sample points from {args.samples}", flush=True)
    fc = build_feature_collection(df)

    out = df.copy()
    for year in args.years:
        print(f"  [EE] sampling {EE_COLLECTION} for {year} ...", flush=True)
        mat = sample_year(fc, year, len(df))
        n_missing = int(np.isnan(mat[:, 0]).sum())
        print(f"       {len(df) - n_missing}/{len(df)} points valid", flush=True)
        cols = [f"emb_{year}_{b:02d}" for b in range(N_DIMS)]
        out[cols] = np.round(mat, 4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} samples -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.2f} MB)")

    # Quick sanity check: a point deforested in 2022 should move much further
    # in embedding space between 2021 and 2023 than a stable forest point.
    y0, y1 = 2021, 2023
    if y0 in args.years and y1 in args.years:
        a = out[[f"emb_{y0}_{b:02d}" for b in range(N_DIMS)]].to_numpy()
        b = out[[f"emb_{y1}_{b:02d}" for b in range(N_DIMS)]].to_numpy()
        dist = np.linalg.norm(a - b, axis=1)
        for label in ("forest", "deforested"):
            m = out["label"] == label
            print(f"  mean |emb_{y0} - emb_{y1}| for {label:12s}: "
                  f"{np.nanmean(dist[m]):.3f}")

    meta = {
        "created": datetime.utcnow().isoformat() + "Z",
        "collection": EE_COLLECTION,
        "years": list(args.years),
        "scale_m": SCALE_M,
        "n_dims": N_DIMS,
        "n_samples": len(out),
        "aligned_with": args.samples,
        "leakage_note": (
            "PRODES target year is 2022; embeddings for 2023-2024 encode the "
            "already-cleared state. The tutorial defaults to years <= 2022 and "
            "uses later years only in an explicit label-leakage exercise."
        ),
        "sources": {
            "embeddings": "AlphaEarth Foundations annual embeddings "
                          "(Google DeepMind), via Google Earth Engine",
            "reference": "Brown et al., 2025, AlphaEarth Foundations",
        },
    }
    meta_path = out_path.parent / "alphaearth_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote provenance -> {meta_path}")


if __name__ == "__main__":
    main()
