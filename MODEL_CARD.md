# Model Card — 1D-CNN Deforestation Classifier (teaching model)

*Following Mitchell et al., "Model Cards for Model Reporting" (2019). This model is
a demonstration built for the CCAI Summer School 2026 "AI for Forestry" tutorial.*

## Model details
- **Type.** Small 1D Convolutional Neural Network (2 Conv1d layers → global average
  pooling → linear head), < 10k parameters, implemented in PyTorch.
- **Input.** A single-channel NDVI time series of ~24 time steps (one year).
- **Output.** A class among `forest`, `deforested`, `old_clearing`.
- **Training.** Cross-entropy loss, Adam optimiser, ~150 full-batch epochs. Trains in
  a few minutes on CPU.
- **Version / date.** v1, 2026, tutorial demonstration.

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
- The model sees only NDVI, so it is blind to forms of change NDVI misses
  (degradation, selective logging, some fire). Extending to more bands/indices and to
  spatial (image) models is discussed in the tutorial's "Next Steps".
