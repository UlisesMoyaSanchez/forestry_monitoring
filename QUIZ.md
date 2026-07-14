# Quiz — Deforestation from Satellite Image Time Series

Six questions assessing the tutorial's key learning objectives. Correct answers and
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

**Q5. Why does the transformer need a *positional encoding* when the 1D CNN does
not?**

- a) To normalise the NDVI values.
- b) Self-attention treats its input as an unordered set — without position
  information, shuffling the time steps would give the same output.
- c) Transformers cannot handle inputs longer than 12 steps.
- d) It reduces the number of parameters.

---

**Q6. Adding the 2023–2024 AlphaEarth embeddings noticeably improves "deforested"
recall for the 2022 labels. Why is this *not* good news for a monitoring system?**

- a) The extra years make training slower.
- b) Embeddings from after the clearing *contain the answer* (label leakage) —
  those features would not exist when a real system had to make the call.
- c) Recall is not a meaningful metric.
- d) The 2023–2024 embeddings are lower resolution.

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
- **Q5 — b.** Self-attention is permutation-invariant: it compares every time step
  with every other but has no built-in notion of order. The (learned) positional
  embedding restores time order. A convolution, by contrast, is inherently local
  and ordered.
- **Q6 — b.** An annual embedding for 2023 or 2024 summarises land that was
  *already cleared*, so the model just reads the answer off the input. Those
  features would not exist at prediction time in a real early-warning system —
  the inflated recall is temporal label leakage, not skill.
