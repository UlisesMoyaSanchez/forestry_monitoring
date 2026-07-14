# Model Card — Deforestation SITS Classifiers (teaching models)

*Following Mitchell et al., "Model Cards for Model Reporting" (2019). These models
are demonstrations built for the CCAI Summer School 2026 "AI for Forestry"
tutorial.*

## Model details
- **`SITSCNN`.** Small 1D Convolutional Neural Network (2 Conv1d layers → global
  average pooling → linear head), < 10k parameters, implemented in PyTorch.
  Input: a single-channel NDVI time series of ~24 time steps (one year).
- **`SITSTransformer`.** Tiny transformer encoder (linear input projection,
  learned positional embedding, 1 self-attention layer with 2 heads,
  `d_model=32`, dropout 0.2, mean-pool → linear head), ~10k parameters. Input:
  any `(batch, T, C)` sequence — used in two configurations:
  1. the NDVI series (`T=24, C=1`), directly comparable to `SITSCNN`;
  2. multi-year **AlphaEarth** annual embeddings (`T=5` years 2018–2022,
     `C=64`), i.e. foundation-model features instead of a hand-crafted index.
- **Output.** A class among `forest`, `deforested`, `old_clearing`.
- **Training.** Cross-entropy loss, Adam optimiser, full-batch epochs (~150–400).
  Each model trains in a few minutes on CPU.
- **Version / date.** v2 (adds the transformer and AlphaEarth inputs), 2026,
  tutorial demonstration.

## Intended use
- **Primary.** Educational — to show how a temporal deep-learning model detects
  deforestation from a vegetation-index time series, and how to evaluate and
  critique it.
- **Users.** CCAI Summer School participants learning AI-for-climate methods.

## Out-of-scope / prohibited use
- **Any real-world decision.** This model must **not** be used for enforcement,
  fines, land-use disputes, credit decisions, trade compliance (e.g. EUDR), or
  operational deforestation mapping.
- It was trained on one small area, one year, one sensor and balanced classes; it
  will not generalise and has not been independently validated.

## Metrics
- Reported on a held-out (spatially separated where possible) test split: accuracy,
  macro-F1, per-class precision/recall, and a confusion matrix. A Random Forest
  baseline is provided for comparison. Exact numbers depend on the run and dataset
  version and are printed in the notebook.

## Training & evaluation data
- The curated Amazon SITS dataset described in `DATASHEET.md` (Sentinel-2 NDVI +
  INPE PRODES labels, southern Pará, PRODES 2022). A synthetic fallback is used only
  when the real CSV cannot be downloaded.
- The AlphaEarth variant uses `data/amazon_alphaearth_samples.csv` (annual 64-d
  Satellite Embedding vectors at the same 908 points; see `DATASHEET.md`).

## Known leakage caveat (AlphaEarth variant)
- The embedding CSV ships years 2018–2024, but the labels refer to the PRODES
  **2022** season. Embeddings for 2023–2024 describe land that was *already
  cleared* and must not be used as input when reporting detection performance —
  the tutorial defaults to years ≤ 2022 and demonstrates the metric inflation
  from post-event years as an explicit label-leakage exercise. Any metric
  obtained with post-event years included does **not** reflect detection skill.

## Ethical considerations & risks
- **Consequences of error.** A false "deforested" prediction could wrongly flag a
  landholder (inspection, credit denial); a missed clearing lets deforestation go
  undetected. Both have real human and environmental stakes.
- **Rarity & imbalance.** Real deforestation is rare; accuracy on balanced teaching
  data overstates real performance.
- **Requirements before real use.** Independent validation against high-resolution
  imagery, expert review of alerts, transparency about accuracy and uncertainty, and
  clear governance over who acts on outputs and how errors are appealed — the kind of
  safeguards INPE builds around PRODES/DETER.

## Caveats
- The NDVI models see only NDVI, so they are blind to forms of change NDVI misses
  (degradation, selective logging, some fire). Extending to more bands/indices and to
  spatial (image) models is discussed in the tutorial's "Next Steps".
- The AlphaEarth features are produced by a proprietary foundation model whose
  training data and failure modes we cannot audit; an embedding dimension cannot be
  explained to an affected landholder the way an NDVI drop can.
