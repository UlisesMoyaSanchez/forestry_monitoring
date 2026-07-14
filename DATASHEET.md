# Datasheet — Amazon SITS Deforestation Samples

*Following Gebru et al., "Datasheets for Datasets" (2021). This dataset was created
for the CCAI Summer School 2026 "AI for Forestry" tutorial.*

## Motivation
- **Purpose.** A small, fast-to-download set of labelled Sentinel-2 NDVI **time
  series** for teaching deforestation detection with a 1D CNN. It is an
  *educational* dataset, not an operational or benchmark product.
- **Created by.** The tutorial authors, for the CCAI Virtual Summer School 2026.
- **Funding.** None specific; built from freely available public data.

## Composition
- **Instances.** Each instance is one ground location, described by an NDVI value
  at ~24 evenly spaced dates across one year, plus a class label and its lon/lat.
- **Classes.** `forest` (stable primary forest), `deforested` (cleared during the
  target PRODES year), `old_clearing` (cleared several years earlier; pasture-like).
- **Count.** A few thousand instances (roughly balanced across classes).
- **Labels.** Derived from INPE **PRODES** *yearly deforestation* polygons.
- **Missing data.** Cloud-obscured observations were removed (via the Sentinel-2
  Scene Classification Layer) and gaps filled by linear temporal interpolation.
- **Sensitive data.** None. All locations are rural land; no personal data.

## Collection process
- **Imagery.** Sentinel-2 L2A surface reflectance (ESA Copernicus), bands red
  (B04) and NIR (B08), accessed via the **Microsoft Planetary Computer** STAC API.
- **Labels.** PRODES `yearly_deforestation_biome` and
  `accumulated_deforestation_2007_biome`, downloaded from INPE's open
  **TerraBrasilis** WFS service.
- **Region / period.** An active deforestation frontier in southern Pará, Brazil;
  PRODES 2022 year (imagery Aug 2021 – Jul 2022).
- **Sampling.** Points sampled inside PRODES polygons (with a small inward buffer to
  avoid mixed edge pixels) for the cleared classes; random points outside all
  recorded clearings for the forest class. NDVI extracted at the nearest pixel.
- **Reproducibility.** Fully reproducible via `data_prep/build_dataset.py`; the
  exact bounding box, dates and temporal grid are recorded in `data/metadata.json`.

## Preprocessing / simplifications (for teaching)
1. Restricted to a small area and a few thousand pixels — **not** representative of
   the whole Amazon or other biomes.
2. Cloud gaps filled by simple linear interpolation, which hides real messiness.
3. Classes deliberately balanced. **Real deforestation is a rare event**; the true
   task suffers severe class imbalance.
4. Only NDVI (from red + NIR) is provided; operational systems use many more bands
   and indices (SWIR, NBR, radar, texture).

## Uses
- **Intended.** Teaching SITS classification, model comparison, evaluation and
  responsible-use reasoning.
- **Out of scope.** Producing real deforestation maps, enforcement, trade
  compliance (e.g. EUDR), scientific benchmarking, or any decision affecting people
  or land. The dataset is too small, too local and too simplified for these.

## Distribution & maintenance
- Distributed with the tutorial repository (CSV + `metadata.json`). No stable DOI
  yet; a Zenodo archive may be added later.
- Best-effort, unmaintained educational artifact. Underlying sources (PRODES,
  Sentinel-2) are maintained by INPE and ESA respectively and should be consulted
  for any real use.

## Known biases & limitations
- Single region, single year, single sensor → **does not generalise**.
- PRODES labels carry their own limitations: 1-ha minimum mapping unit, annual
  cadence, focus on clear-cut (degradation only partially captured).
- Balanced classes misrepresent the true rarity of deforestation.

---

# Companion dataset — AlphaEarth embeddings (`amazon_alphaearth_samples.csv`)

- **What it is.** Annual 64-dimensional embedding vectors from Google's
  **Satellite Embedding** dataset (`GOOGLE/SATELLITE_EMBEDDING/V1_ANNUAL`,
  AlphaEarth Foundations; Brown et al. 2025), sampled at 10 m for the **same 908
  points** as the NDVI CSV, years **2018–2024**. Wide format:
  `lon, lat, label, emb_2018_00 … emb_2024_63` (451 columns), values in [−1, 1]
  rounded to 4 decimals. Rows are aligned one-to-one with
  `amazon_sits_samples.csv`.
- **Collection.** Extracted once by the tutorial authors with
  `data_prep/build_alphaearth.py` via the Google Earth Engine API
  (`sampleRegions`, scale = 10 m); provenance in `data/alphaearth_metadata.json`.
  Participants do not need an Earth Engine account.
- **Derived data caveat.** These are *learned features* from a proprietary
  foundation model trained on multi-sensor data; their content cannot be fully
  audited or attributed to specific observations.
- **Known issue — temporal label leakage.** The class labels refer to the PRODES
  **2022** season. Embeddings for **2023–2024 describe land that was already
  cleared** and therefore encode the label. They are included *deliberately* for
  a leakage exercise; any model evaluated with post-event years as input does
  not measure detection skill. The tutorial's default uses years ≤ 2022 only.
- **Missing data.** Points falling in masked pixels for a year yield empty
  (NaN) embedding columns and are dropped consistently in the notebook.
