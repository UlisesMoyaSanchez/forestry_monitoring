#!/usr/bin/env python3
"""Extra AlphaEarth-only teaching figures for the CCAI deforestation-SITS
tutorial. Companion to make_figures.py -- kept separate because these depend on
the *temporal trajectory* of the embeddings (2018-2024), which is what makes a
foundation model different from a single-year classifier.

  figures/fig_09_alphaearth_drift.png   mean year-over-year embedding
                                        displacement per class (WHEN change happens)
  figures/anim_04_alphaearth_pca.gif    a fixed 2-D PCA projection animated across
                                        years -- deforested pixels drift out of the
                                        forest cluster into the clearing cluster

Both reuse the palette/helpers from make_figures.py. They need the real CSV
(data/amazon_alphaearth_samples.csv); with the synthetic fallback they are
watermarked as placeholders.

Run:  python data_prep/make_alphaearth_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (  # noqa: E402  (shared style + data loaders)
    AE_DIMS, CLASS_STYLE, GRID, INK, INK2, MUTED, OUT, SURFACE,
    clean, load, load_alphaearth, synth_watermark,
)

ALL_YEARS = list(range(2018, 2025))  # full trajectory, incl. post-deforestation


def _year_matrix(ae_df, year):
    """(n_valid, 64) embeddings for one year, plus the matching label array."""
    cols = [f"emb_{year}_{b:02d}" for b in range(AE_DIMS)]
    valid = ae_df[cols].notna().all(axis=1)
    X = ae_df.loc[valid, cols].to_numpy("float32")
    labels = ae_df.loc[valid, "label"].to_numpy()
    return X, labels


def fig_alphaearth_drift(ae_df, order, is_synth, years=ALL_YEARS):
    """Mean L2 displacement between consecutive years, per class.

    A pixel that is cleared moves a long way in embedding space the year the
    forest disappears; stable forest and old pasture barely move. This puts a
    number on the sanity check printed by build_alphaearth.py.
    """
    pair_labels = [f"{y}→{y + 1}" for y in years[:-1]]
    x = np.arange(len(pair_labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    clean(ax)
    for cls in order:
        st = CLASS_STYLE[cls]
        means = []
        for y in years[:-1]:
            Xa, la = _year_matrix(ae_df, y)
            Xb, lb = _year_matrix(ae_df, y + 1)
            m = (la == cls)
            d = np.linalg.norm(Xa[m] - Xb[m], axis=1)
            means.append(float(np.nanmean(d)) if d.size else np.nan)
        ax.plot(x, means, color=st["color"], lw=2.4, marker=st["marker"],
                ms=8, mfc=st["color"], mec=SURFACE, mew=1.4, label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("consecutive-year pair")
    ax.set_ylabel("mean |Δ embedding|  (L2 distance)")
    ax.set_title("How far each pixel moves in AlphaEarth space, year to year\n"
                 "the clearing year spikes for deforested pixels; stable land stays put",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.95,
              edgecolor=GRID)
    if is_synth:
        synth_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig_09_alphaearth_drift.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_09_alphaearth_drift.png")


def anim_alphaearth_pca(ae_df, order, is_synth, years=ALL_YEARS):
    """Animate a *fixed* 2-D PCA projection across years.

    The projection is fit once on all years pooled, so the axes mean the same
    thing in every frame and motion is real motion, not a re-fit wobble.
    """
    # Fit the projection on every (point, year) embedding stacked together.
    stack = np.vstack([_year_matrix(ae_df, y)[0] for y in years])
    mean = stack.mean(0)
    _, _, vt = np.linalg.svd(stack - mean, full_matrices=False)
    basis = vt[:2].T  # (64, 2)

    # Pre-project every year so we can fix the axis limits up front.
    proj = {}
    labels_ref = None
    for y in years:
        X, labels = _year_matrix(ae_df, y)
        proj[y] = (X - mean) @ basis
        labels_ref = labels
    allpts = np.vstack(list(proj.values()))
    pad = 0.05 * (allpts.max(0) - allpts.min(0))
    xlim = (allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0])
    ylim = (allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    clean(ax)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    scatters = {}
    for cls in order:
        st = CLASS_STYLE[cls]
        scatters[cls] = ax.scatter([], [], s=34, marker=st["marker"],
                                   c=st["color"], edgecolors=SURFACE,
                                   linewidths=0.5, alpha=0.85, label=st["label"])
    ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.95,
              edgecolor=GRID)
    title = ax.set_title("", fontsize=12.5, pad=10, color=INK)
    # Reserve bottom margin so the caption sits inside the (fixed) frame; gif
    # writers discard bbox_inches, so nothing outside the figure would show.
    fig.subplots_adjust(bottom=0.14)
    fig.text(0.5, 0.03,
             "same projection every frame  ·  watch the red X's leave the forest",
             ha="center", fontsize=9.5, color=INK2)
    if is_synth:
        synth_watermark(fig)

    # Slow down on the clearing year so the eye catches the jump.
    seq = []
    for y in years:
        seq.append(y)
        if y == 2022:
            seq += [y, y]
    seq += [years[-1], years[-1]]  # hold on the final state

    def frame(k):
        y = seq[k]
        pts = proj[y]
        for cls in order:
            m = labels_ref == cls
            scatters[cls].set_offsets(pts[m])
        highlight = "  ◀ clearing year" if y == 2022 else ""
        title.set_text(f"AlphaEarth embeddings drifting through time — {y}{highlight}")
        return list(scatters.values()) + [title]

    anim = FuncAnimation(fig, frame, frames=len(seq), interval=650, blit=False)
    anim.save(OUT / "anim_04_alphaearth_pca.gif", writer=PillowWriter(fps=1.6))
    plt.close(fig)
    print("  wrote anim_04_alphaearth_pca.gif")


def main():
    df, _, order = load()
    ae_df, is_synth = load_alphaearth(df)
    if is_synth:
        print("  (AlphaEarth CSV not found: figures use watermarked synthetic data)")
    print("9) AlphaEarth year-over-year drift ...")
    fig_alphaearth_drift(ae_df, order, is_synth)
    print("A) AlphaEarth PCA drift animation ...")
    anim_alphaearth_pca(ae_df, order, is_synth)


if __name__ == "__main__":
    main()
