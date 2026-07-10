# Quiz — Deforestation from Satellite Image Time Series

Four questions assessing the tutorial's key learning objectives. Correct answers and
explanations are at the bottom.

---

**Q1. Why classify a *time series* of satellite images rather than a single-date image
for tropical deforestation?**

- a) Time series are smaller files.
- b) Persistent clouds make any single date unreliable, and a clearing is only
  recognisable as a *drop over time*.
- c) Single images cannot be used to compute NDVI.
- d) Time series remove the need for labels.

---

**Q2. (True / False)** *"A false-positive deforestation prediction under the EU
Deforestation Regulation is harmless, because a human always checks before any action
is taken."*

---

**Q3. What problem does a *spatially-aware* train/test split protect against?**

- a) Overfitting to the optimiser's learning rate.
- b) GPU out-of-memory errors.
- c) Leakage from spatial autocorrelation, which would inflate the test accuracy.
- d) Cloud contamination of the imagery.

---

**Q4. In this tutorial's model, what does a 1D convolution operate over?**

- a) The spatial x–y plane of the image.
- b) The time axis of the NDVI series.
- c) The red/green/blue colour channels.
- d) The list of class labels.

---

## Answer key

- **Q1 — b.** In the cloudy tropics a single date is often unusable; the *temporal
  trajectory* (a high plateau followed by a permanent drop) is what identifies a
  clearing.
- **Q2 — False.** A false positive can wrongly block trade or credit; human review,
  validation and governance are exactly what must be in place *because* the model can
  be wrong. It is not harmless by default.
- **Q3 — c.** Nearby pixels are correlated (even from the same clearing); a random
  split leaks near-duplicates into the test set and overestimates performance.
- **Q4 — b.** The 1D convolution slides filters along the **time** axis to detect
  temporal patterns such as plateaus and drops.
