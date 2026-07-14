# Detecting Deforestation from Satellite Image Time Series

A hands-on tutorial for the **[Climate Change AI](https://www.climatechange.ai/)
Virtual Summer School 2026** — module *AI for Forestry*.

Participants learn to detect tropical deforestation by treating each ground
location as a **time series** of a vegetation index (NDVI) across one year, and
training models to tell *stable forest*, *freshly deforested*, and *old clearing*
trajectories apart. The workflow uses real **Sentinel-2** imagery and official
**INPE PRODES** deforestation labels over an active frontier in the Brazilian
Amazon, then extends to **AlphaEarth foundation-model embeddings** (Google's
Satellite Embedding dataset) read by a small **time series Transformer**.

> **Educational use only.** Nothing here is fit for operational deforestation
> mapping, enforcement, trade compliance (e.g. EUDR), or any decision affecting
> people or land. See [Responsible use](#responsible-use).

---

## Learning objectives

By the end of the tutorial you can:

1. **Explain** why deforestation monitoring matters for the climate and who uses it.
2. **Describe** what a Satellite Image Time Series (SITS) is and why time series
   beat a single (often cloudy) image.
3. **Build** a reproducible labelled dataset from open satellite imagery and
   official labels.
4. **Train and compare** a classical baseline (Random Forest), a 1D
   Convolutional Neural Network and a time series Transformer.
5. **Compare** hand-crafted features (NDVI) with foundation-model embeddings
   (AlphaEarth) — and recognise the temporal label-leakage trap that annual
   embeddings set.
6. **Evaluate honestly** — read per-class metrics and a confusion matrix, not just
   accuracy — and reason about limitations and responsible use.

---

## Quickstart

### Run in Google Colab (recommended, no setup)

Open `CCAI_Tutorial_Deforestation_SITS.ipynb` in Colab and run the cells top to
bottom. The curated dataset (~0.5 MB) downloads automatically; if the download
fails, the notebook falls back to a clearly-labelled synthetic dataset so the rest
still runs. Estimated time: **under 2 hours**, CPU is enough.

### Run locally

```bash
# option A: pip
pip install -r requirements.txt
jupyter lab CCAI_Tutorial_Deforestation_SITS.ipynb

# option B: conda (also includes the geospatial stack for rebuilding data)
conda env create -f environment.yml
conda activate forestry-ccai
jupyter lab CCAI_Tutorial_Deforestation_SITS.ipynb
```

Smoke-test the whole pipeline (imports, data load, all three model families,
AlphaEarth embeddings, CodeCarbon):

```bash
python check_pipeline.py     # prints PASS/FAIL for each stage
```

---

## Repository structure

| Path | What it is |
|------|------------|
| `CCAI_Tutorial_Deforestation_SITS.ipynb` | **The tutorial notebook** — the main deliverable. |
| `data/amazon_sits_samples.csv` | Curated dataset: 908 labelled NDVI time series (24 steps). |
| `data/metadata.json` | Exact provenance (bbox, dates, temporal grid, sources). |
| `data/amazon_alphaearth_samples.csv` | AlphaEarth annual embeddings (64-d × 2018–2024) at the same 908 points.* |
| `data/alphaearth_metadata.json` | Provenance for the embedding CSV.* |
| `data_prep/build_dataset.py` | Creator-side script that rebuilds the dataset from source. |
| `data_prep/build_alphaearth.py` | Creator-side script that extracts the AlphaEarth embeddings (needs Earth Engine auth). |
| `data_prep/make_figures.py` | Generates all figures & GIFs from the real data + pipeline. |
| `figures/` | Static figures (PNG) and animations (GIF) — see below. |
| `slides/deforestation_sits.tex` / `.pdf` | Companion LaTeX Beamer deck explaining every step. |
| `check_pipeline.py` | End-to-end smoke test. |
| `DATASHEET.md` | Dataset documentation (Gebru et al. 2021). |
| `MODEL_CARD.md` | Model documentation (Mitchell et al. 2019). |
| `QUIZ.md` | Comprehension quiz questions. |
| `requirements.txt` / `environment.yml` | Pinned dependencies (pip / conda). |

\* Not yet committed: run `data_prep/build_alphaearth.py` once (see below). Until
then the notebook's Section 9 runs on a clearly-flagged synthetic fallback.

---

## The data

Each row of `data/amazon_sits_samples.csv` is one ground location described by
`lon, lat, label` and 24 NDVI values (`ndvi_00 … ndvi_23`) evenly spaced across
**Aug 2021 – Jul 2022** (the PRODES 2022 reporting year).

- **Imagery** — Sentinel-2 L2A surface reflectance (ESA Copernicus), bands B04
  (red) and B08 (NIR) for NDVI, plus the Scene Classification Layer for cloud
  masking, accessed via the **Microsoft Planetary Computer** STAC API.
- **Labels** — INPE **PRODES** yearly / accumulated deforestation polygons from
  the open **TerraBrasilis** WFS service.
- **Region** — a ~22×22 km window on the BR-163 / Rio Novo arc, southern Pará,
  Brazil (`bbox = [-54.75, -3.70, -54.55, -3.50]`).
- **Classes** — `forest` (stable), `deforested` (cleared this year),
  `old_clearing` (cleared years ago, pasture-like).

Full details and known biases are in **[DATASHEET.md](DATASHEET.md)**.

### Rebuild the dataset from source

The CSV is committed so participants never need to. To regenerate it (requires the
`forestry-ccai` conda env for the geospatial stack):

```bash
python data_prep/build_dataset.py --out data/amazon_sits_samples.csv
```

This downloads labels + imagery, cloud-masks via SCL, builds a **maximum-value
composite** on a regular 15-day grid, gap-fills, samples labelled pixels, and
writes the CSV plus `metadata.json`. Expect several minutes (network-bound).

### Build the AlphaEarth embedding CSV (creator-side, once)

Section 9 of the notebook uses annual 64-dimensional embeddings from Google's
[Satellite Embedding dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
(AlphaEarth Foundations), sampled at the same 908 points for years 2018–2024.
Because the dataset lives in Google Earth Engine (which needs an account), the
embeddings are pre-extracted once and committed, so participants stay auth-free:

```bash
pip install earthengine-api
earthengine authenticate                       # one time
python data_prep/build_alphaearth.py --project <your-gcloud-project>
```

This writes `data/amazon_alphaearth_samples.csv` (wide format:
`lon, lat, label, emb_2018_00 … emb_2024_63`) plus `data/alphaearth_metadata.json`,
and prints a sanity check (a deforested point should move much further in
embedding space between 2021 and 2023 than a forest point).

> **Leakage note (by design).** The CSV includes 2023–2024 on purpose: the
> notebook defaults to years ≤ 2022 and uses the post-event years only in an
> explicit label-leakage exercise.

---

## Figures & animations

Generated by `data_prep/make_figures.py`, which runs the notebook's **real**
pipeline (spatial split + the actual `SITSCNN`, 150 epochs) so nothing is mocked.

| File | Stage |
|------|-------|
| `fig_01_data_input.png` · `anim_01_ndvi_reveal.gif` | Input NDVI time series per class |
| `fig_02_labels.png` | Labelled pixels & the spatial train/test split |
| `fig_03_model.png` | The 1D-CNN architecture |
| `anim_03_training.gif` · `fig_03b_training.png` | Training loss & accuracy |
| `fig_04_confusion.png` | Test-set confusion matrix |
| `fig_05_inference_map.png` | Predictions across the held-out region |
| `fig_06_transformer.png` | The `SITSTransformer` architecture |
| `fig_07_model_comparison.png` | 4-model metric comparison (grouped bars)† |
| `fig_08_alphaearth_pca.png` | 2-D PCA of the AlphaEarth embeddings† |

† Watermarked as synthetic placeholders until `build_alphaearth.py` has produced
the real embedding CSV — re-run `make_figures.py` afterwards.

Regenerate them (needs `matplotlib`, `torch`, `pillow`):

```bash
python data_prep/make_figures.py
```

Class colours (forest = green, deforested = red, old clearing = yellow) are
colourblind-checked, and each class also carries a distinct marker shape so
identity never rests on colour alone.

---

## Method & results

Four models are trained and compared on a **spatial** train/test split (eastern 30%
of the window held out, to avoid leaking correlated neighbouring pixels):

- **Baseline** — Random Forest on the 24 NDVI values as flat features.
- **Deep model 1** — `SITSCNN`, a small 1D CNN (two `Conv1d` layers → global average
  pooling → linear head, <10k parameters) whose convolutions slide along the time
  axis to detect shapes like "a mid-year drop".
- **Deep model 2** — `SITSTransformer`, a tiny transformer encoder (learned
  positional embedding, 1 attention layer, ~10k parameters) whose input
  `(batch, T, C)` is deliberately shape-agnostic.
- **Deep model 2, new data** — the *same* `SITSTransformer` on the multi-year
  AlphaEarth embedding series (5 years × 64 dims, leakage-safe default ≤ 2022).

| Model | Accuracy | Macro-F1 | Deforested recall |
|-------|:--------:|:--------:|:-----------------:|
| Random Forest (baseline) | 0.68 | **0.59** | **0.39** |
| 1D CNN | 0.64 | 0.41 | 0.00 |
| Transformer (NDVI) | **0.69** | 0.58 | 0.22 |
| Transformer (AlphaEarth) | — | — | — |

*(AlphaEarth numbers pending the real embedding CSV; the notebook prints them
live once it is built.)*

> ⚠️ **The honest result is the lesson.** On this real, cloudy dataset the simple
> baseline still finds the most actual deforestation, the CNN collapses to the
> majority class — detecting **none** of the deforested pixels — and the
> transformer lands in between. Wet-season cloud drags even stable forest's NDVI
> down, so forest and fresh clearings barely separate on NDVI alone. This is a
> deliberate teaching case in **class imbalance**, **weak signals**, and **why
> accuracy alone misleads** — read the confusion matrix, compare against a
> baseline, and look at per-class recall. Strategies to improve it (class
> weighting, more bands/indices, a cleaner time window, a spatial model) are
> discussed in the notebook's *Next Steps*. Section 9 adds a second lesson:
> post-event AlphaEarth years boost recall through **temporal label leakage**,
> not skill.

Numbers depend on the run and dataset version and are printed live in the notebook.
See **[MODEL_CARD.md](MODEL_CARD.md)** for the full model documentation.

---

## Responsible use

This is a **teaching model on a tiny, local, single-year, single-sensor dataset
with balanced classes** — it does not generalise and has not been independently
validated.

- **A false "deforested"** could wrongly flag a landholder (inspection, credit
  denial); **a missed clearing** lets destruction go undetected. Both carry real
  human and environmental stakes.
- Before any real use you would need independent validation against high-resolution
  imagery, expert review of alerts, transparency about accuracy and uncertainty,
  and clear governance over who acts on outputs and how errors are appealed — the
  kind of safeguards INPE builds around PRODES / DETER.
- **Out of scope:** operational mapping, enforcement, fines, land-use disputes,
  credit decisions, trade compliance (e.g. EUDR).

---

## Best-practice instrumentation

Training is wrapped in **[CodeCarbon](https://codecarbon.io/)** to log energy use
and CO₂ emissions — negligible for this toy model, but a reminder that training at
national scale, repeatedly, is not.

---

## Sources & further reading

- IPCC AR6 WGIII, Ch. 7 (AFOLU); Ometto et al., 2022.
- INPE PRODES & DETER — <https://terrabrasilis.dpi.inpe.br>.
- ESA Copernicus Sentinel-2; Microsoft Planetary Computer.
- Gebru et al., *Datasheets for Datasets*, 2021.
- Mitchell et al., *Model Cards for Model Reporting*, 2019.
- EU Deforestation Regulation (EU) 2023/1115.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Sainte Fare Garnot & Landrieu, *Lightweight Temporal Self-Attention for
  Classifying Satellite Image Time Series*, 2020.
- Brown et al., *AlphaEarth Foundations*, 2025 — [Satellite Embedding dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).

---

*Built for the CCAI Virtual Summer School 2026. Data © ESA (Copernicus) and INPE
(PRODES), used under their open terms.*
