#!/usr/bin/env python3
"""Generate teaching figures and GIFs for the CCAI deforestation-SITS tutorial.

CREATOR-SIDE SCRIPT. Produces static PNGs and animated GIFs that illustrate the
five stages of the workflow, using the *real* curated dataset and the *same*
model/pipeline the notebook runs:

  figures/fig_01_data_input.png     NDVI time series, one panel per class
  figures/anim_01_ndvi_reveal.gif   a single trajectory revealed step-by-step
  figures/fig_02_labels.png         where the labelled pixels come from (map)
  figures/fig_03_model.png          the 1D-CNN architecture, box by box
  figures/anim_03_training.gif      loss + test-accuracy filling in per epoch
  figures/fig_04_confusion.png      test-set confusion matrix
  figures/fig_05_inference_map.png  predictions across the test region
  figures/fig_06_transformer.png    the SITSTransformer architecture
  figures/fig_07_model_comparison.png  4-model metric comparison (grouped bars)
  figures/fig_08_alphaearth_pca.png    2-D PCA of the AlphaEarth embeddings

The AlphaEarth figures use data/amazon_alphaearth_samples.csv when it exists
(built by build_alphaearth.py); otherwise they fall back to synthetic
embeddings and are watermarked as placeholders -- re-run this script after
building the real CSV.

Run:  python data_prep/make_figures.py
Requires the `forestry-ccai` env (matplotlib, numpy, pandas, torch, pillow).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
DATA_CSV = REPO / "data" / "amazon_sits_samples.csv"
AE_CSV = REPO / "data" / "amazon_alphaearth_samples.csv"
AE_YEARS = list(range(2018, 2023))  # leakage-safe default, like the notebook
AE_DIMS = 64
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)
SEED = 42

# --- validated palette (dataviz skill; classes = categorical identity) -------
# forest=green, deforested=red, old_clearing=yellow. Worst CVD dE=12.3 (>=12
# target). Marker SHAPE is a second identity channel so colour is never alone.
CLASS_STYLE = {
    "forest":       dict(color="#008300", marker="o", label="Forest (stable)"),
    "deforested":   dict(color="#e34948", marker="X", label="Deforested (this year)"),
    "old_clearing": dict(color="#eda100", marker="^", label="Old clearing (pasture)"),
}
SEQ_BLUE = ["#eef4fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
SURFACE = "#fcfcfb"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})


def clean(ax):
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load():
    df = pd.read_csv(DATA_CSV)
    ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
    order = [c for c in CLASS_STYLE if c in df["label"].unique()]
    return df, ndvi_cols, order


# =============================================================================
# 1. DATA INPUT -- NDVI time series per class + reveal GIF
# =============================================================================
def fig_data_input(df, ndvi_cols, order):
    T = len(ndvi_cols)
    x = np.arange(T)
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.8),
                             sharey=True)
    if len(order) == 1:
        axes = [axes]
    rng = np.random.default_rng(SEED)
    for ax, cls in zip(axes, order):
        st = CLASS_STYLE[cls]
        sub = df[df["label"] == cls][ndvi_cols].to_numpy("float32")
        idx = rng.choice(len(sub), size=min(25, len(sub)), replace=False)
        for row in sub[idx]:
            ax.plot(x, row, color=st["color"], lw=1.0, alpha=0.15)
        ax.plot(x, sub.mean(0), color=st["color"], lw=2.5,
                marker=st["marker"], ms=6, mfc=st["color"], mec=SURFACE, mew=1.5)
        ax.set_title(st["label"], fontsize=11, pad=8)
        ax.set_xlabel("time step (~15-day composite)")
        ax.set_ylim(-0.05, 1.0)
        clean(ax)
    axes[0].set_ylabel("NDVI  (greenness)")
    fig.suptitle("Input data: one year of NDVI per pixel  ·  thin = samples, bold = class mean",
                 fontsize=12.5, y=1.02, x=0.5, ha="center", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_01_data_input.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_01_data_input.png")


def anim_ndvi_reveal(df, ndvi_cols, order):
    """A single deforested trajectory revealed step by step, with a moving
    'satellite pass' marker -- shows what one training example actually is."""
    T = len(ndvi_cols)
    x = np.arange(T)
    cls = "deforested" if "deforested" in order else order[0]
    st = CLASS_STYLE[cls]
    sub = df[df["label"] == cls][ndvi_cols].to_numpy("float32")
    # pick a clear example: biggest early->late NDVI drop
    drop = sub[:, :T // 3].mean(1) - sub[:, -T // 3:].mean(1)
    series = sub[np.argsort(-drop)[3]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.13)
    clean(ax)
    ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("time step (~15-day composite over one year)")
    ax.set_ylabel("NDVI  (greenness)")
    ax.set_title("One training example: NDVI of a single pixel through the year",
                 fontsize=12, pad=10)
    ax.axhspan(-0.05, 0.3, color=st["color"], alpha=0.05)
    ax.text(T - 0.6, 0.02, "bare / cleared", ha="right", va="bottom",
            fontsize=9, color=INK2)

    (line,) = ax.plot([], [], color=st["color"], lw=2.5)
    (head,) = ax.plot([], [], marker=st["marker"], ms=13, color=st["color"],
                      mec=SURFACE, mew=2, ls="none")
    label = ax.text(0.02, 0.06, "", transform=ax.transAxes, fontsize=10,
                    color=INK2)

    hold = 8  # extra frames at the end so the GIF pauses on the full curve

    def frame(k):
        i = min(k, T - 1)
        line.set_data(x[:i + 1], series[:i + 1])
        head.set_data([x[i]], [series[i]])
        label.set_text(f"observations so far: {i + 1}/{T}")
        return line, head, label

    anim = FuncAnimation(fig, frame, frames=T + hold, interval=180, blit=True)
    anim.save(OUT / "anim_01_ndvi_reveal.gif", writer=PillowWriter(fps=6))
    plt.close(fig)
    print("  wrote anim_01_ndvi_reveal.gif")


# =============================================================================
# 2. LABELS -- where the labelled pixels come from (spatial map)
# =============================================================================
def fig_labels(df, ndvi_cols, order):
    if not {"lon", "lat"}.issubset(df.columns):
        print("  (skip labels map: no lon/lat columns)")
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    thresh = df["lon"].quantile(0.7)  # matches notebook's spatial split
    for cls in order:
        st = CLASS_STYLE[cls]
        sub = df[df["label"] == cls]
        ax.scatter(sub["lon"], sub["lat"], s=34, marker=st["marker"],
                   c=st["color"], edgecolors=SURFACE, linewidths=0.6,
                   alpha=0.9, label=st["label"])
    ax.axvline(thresh, color=INK2, lw=1.5, ls=(0, (5, 4)))
    ymax = df["lat"].max()
    ax.text(thresh - 0.001, ymax, "train  ", ha="right", va="top",
            fontsize=10, color=INK2, style="italic")
    ax.text(thresh + 0.001, ymax, "  test", ha="left", va="top",
            fontsize=10, color=INK2, style="italic")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Labelled pixels & the spatial train/test split\n"
                 "PRODES polygons (INPE) label each Sentinel-2 pixel",
                 fontsize=12, pad=10)
    ax.set_aspect("equal", adjustable="datalim")
    clean(ax)
    ax.legend(loc="lower left", frameon=True, fontsize=9, framealpha=0.95,
              edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig_02_labels.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_02_labels.png")


# =============================================================================
# 3. MODEL -- 1D-CNN architecture diagram
# =============================================================================
def fig_model(T, n_classes):
    blocks = [
        ("Input\nNDVI series", f"1 x {T}", "#eef4fd"),
        ("Conv1d\nk=5, ReLU", f"16 x {T}", SEQ_BLUE[2]),
        ("Conv1d\nk=5, ReLU", f"32 x {T}", SEQ_BLUE[3]),
        ("Global\navg-pool", "32 x 1", SEQ_BLUE[4]),
        ("Linear", f"{n_classes}", SEQ_BLUE[5]),
        ("Softmax\nclass prob.", f"{n_classes}", "#008300"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, len(blocks)); ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, y0 = 0.72, 0.42, 0.30
    for i, (name, shape, col) in enumerate(blocks):
        x = i + (1 - w) / 2
        dark = col in (SEQ_BLUE[4], SEQ_BLUE[5], "#008300")
        txt = "#ffffff" if dark else INK
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                             linewidth=0, facecolor=col)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, name, ha="center", va="center",
                fontsize=10, color=txt)
        ax.text(x + w / 2, y0 - 0.09, shape, ha="center", va="center",
                fontsize=9, color=INK2, family="monospace")
        if i < len(blocks) - 1:
            arr = FancyArrowPatch((x + w + 0.02, y0 + h / 2),
                                  (i + 1 + (1 - w) / 2 - 0.02, y0 + h / 2),
                                  arrowstyle="-|>", mutation_scale=13,
                                  color=MUTED, lw=1.5)
            ax.add_patch(arr)
    ax.text(len(blocks) / 2, 0.90, "SITSCNN  ·  a small 1D convolutional network over time",
            ha="center", fontsize=13, color=INK, weight="bold")
    ax.text(len(blocks) / 2, 0.06,
            "convolutions slide along the TIME axis, learning shapes like "
            "“a mid-year drop”; global pooling makes it length-robust",
            ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_model.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_03_model.png")


# =============================================================================
# Train the SAME models the notebook trains (for real training/inference figs)
# =============================================================================
def split_mask(df):
    """The notebook's spatial split (eastern 30% of longitudes = test)."""
    if {"lon", "lat"}.issubset(df.columns):
        thresh = df["lon"].quantile(0.7)
        return (df["lon"] >= thresh).to_numpy()
    rng = np.random.default_rng(SEED)
    return rng.random(len(df)) < 0.3


def build_cnn(n_classes):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
        nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(32, n_classes),
    )


def build_transformer(n_classes, in_dim, seq_len,
                      d_model=32, n_heads=2, n_layers=1, dropout=0.2):
    """The notebook's SITSTransformer: input (batch, T, C)."""
    import torch
    import torch.nn as nn

    class SITSTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=2 * d_model,
                dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            h = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
            return self.head(self.encoder(h).mean(dim=1))

    return SITSTransformer()


def train_torch_model(model, X3d, y, test_mask, epochs=150, lr=3e-3):
    """Full-batch training on already-shaped tensors; returns (hist, yte, pred)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    Xtr_t = torch.tensor(X3d[~test_mask])
    Xte_t = torch.tensor(X3d[test_mask])
    ytr_t = torch.tensor(y[~test_mask], dtype=torch.long)
    yte_t = torch.tensor(y[test_mask], dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"loss": [], "acc": []}
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
        hist["loss"].append(loss.item()); hist["acc"].append(acc)
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    return hist, y[test_mask], pred


def train_cnn(df, ndvi_cols, order):
    np.random.seed(SEED)
    cls2idx = {c: i for i, c in enumerate(order)}
    X = df[ndvi_cols].to_numpy("float32")
    y = df["label"].map(cls2idx).to_numpy()
    test_mask = split_mask(df)
    hist, yte, pred = train_torch_model(
        build_cnn(len(order)), X[:, None, :], y, test_mask)
    return hist, yte, pred, X[test_mask]


def anim_training(hist, n_classes):
    ep = np.arange(1, len(hist["loss"]) + 1)
    loss = np.array(hist["loss"]); acc = np.array(hist["acc"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for a in (a0, a1):
        clean(a); a.set_xlabel("epoch"); a.set_xlim(0, len(ep))
    a0.set_title("Training loss", fontsize=11); a0.set_ylabel("cross-entropy")
    a0.set_ylim(0, max(loss) * 1.05)
    a1.set_title("Test accuracy", fontsize=11); a1.set_ylabel("accuracy")
    a1.set_ylim(0, 1.02)
    a1.axhline(1 / n_classes, ls=(0, (5, 4)), color=MUTED, lw=1.3)
    a1.text(len(ep) * 0.98, 1 / n_classes + 0.02, "chance", ha="right",
            fontsize=9, color=MUTED)
    (l0,) = a0.plot([], [], color="#e34948", lw=2.2)
    (d0,) = a0.plot([], [], marker="o", ms=8, color="#e34948", mec=SURFACE, mew=2, ls="none")
    (l1,) = a1.plot([], [], color="#2a78d6", lw=2.2)
    (d1,) = a1.plot([], [], marker="o", ms=8, color="#2a78d6", mec=SURFACE, mew=2, ls="none")
    t1 = a1.text(0.04, 0.08, "", transform=a1.transAxes, fontsize=11, color=INK)
    fig.suptitle("Training the 1D-CNN over 150 epochs", fontsize=12.5, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.13, wspace=0.22)

    step = 3  # frames every 3 epochs to keep the GIF light
    ks = list(range(0, len(ep), step)) + [len(ep) - 1] * 8

    def frame(k):
        l0.set_data(ep[:k + 1], loss[:k + 1]); d0.set_data([ep[k]], [loss[k]])
        l1.set_data(ep[:k + 1], acc[:k + 1]); d1.set_data([ep[k]], [acc[k]])
        t1.set_text(f"epoch {ep[k]:3d}   acc = {acc[k]:.2f}")
        return l0, d0, l1, d1, t1

    anim = FuncAnimation(fig, frame, frames=ks, interval=90, blit=True)
    anim.save(OUT / "anim_03_training.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print("  wrote anim_03_training.gif")


def fig_confusion(yte, pred, order):
    n = len(order)
    cm = np.zeros((n, n), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    short = {"forest": "Forest", "deforested": "Deforested",
             "old_clearing": "Old clearing"}
    labs = [short.get(c, c) for c in order]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, fontsize=9); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)\nrow-normalised; diagonal = correct",
                 fontsize=12, pad=10)
    for i in range(n):
        for j in range(n):
            val = cmn[i, j]
            ax.text(j, i, f"{val:.0%}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if val > 0.5 else INK)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_04_confusion.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_04_confusion.png")


def fig_inference_map(df, pred, yte, order):
    if not {"lon", "lat"}.issubset(df.columns):
        print("  (skip inference map: no lon/lat columns)")
        return
    thresh = df["lon"].quantile(0.7)
    test = df[df["lon"] >= thresh].reset_index(drop=True)
    if len(test) != len(pred):
        print("  (skip inference map: test size mismatch)")
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for ci, cls in enumerate(order):
        st = CLASS_STYLE[cls]
        m = pred == ci
        ax.scatter(test.loc[m, "lon"], test.loc[m, "lat"], s=42,
                   marker=st["marker"], c=st["color"], edgecolors=SURFACE,
                   linewidths=0.6, alpha=0.9, label=f"predicted {st['label']}")
    wrong = pred != yte
    ax.scatter(test.loc[wrong, "lon"], test.loc[wrong, "lat"], s=150,
               facecolors="none", edgecolors=INK, linewidths=1.6,
               label="misclassified", zorder=5)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    acc = (pred == yte).mean()
    ax.set_title(f"Inference on the held-out test region  ·  accuracy = {acc:.0%}\n"
                 "black rings mark the model's mistakes", fontsize=12, pad=10)
    ax.set_aspect("equal", adjustable="datalim")
    clean(ax)
    ax.legend(loc="lower left", frameon=True, fontsize=8.5, framealpha=0.95,
              edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig_05_inference_map.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_05_inference_map.png")


# =============================================================================
# 6. TRANSFORMER -- SITSTransformer architecture diagram
# =============================================================================
def fig_transformer(T, n_classes):
    blocks = [
        ("Input\ntime series", f"T x C", "#eef4fd"),
        ("Linear proj.\n+ pos. embed", "T x 32", SEQ_BLUE[2]),
        ("Self-attention\nencoder x1", "T x 32", SEQ_BLUE[3]),
        ("Mean-pool\nover time", "32", SEQ_BLUE[4]),
        ("Linear", f"{n_classes}", SEQ_BLUE[5]),
        ("Softmax\nclass prob.", f"{n_classes}", "#008300"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, len(blocks)); ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, y0 = 0.72, 0.42, 0.30
    for i, (name, shape, col) in enumerate(blocks):
        x = i + (1 - w) / 2
        dark = col in (SEQ_BLUE[4], SEQ_BLUE[5], "#008300")
        txt = "#ffffff" if dark else INK
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                             linewidth=0, facecolor=col)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, name, ha="center", va="center",
                fontsize=10, color=txt)
        ax.text(x + w / 2, y0 - 0.09, shape, ha="center", va="center",
                fontsize=9, color=INK2, family="monospace")
        if i < len(blocks) - 1:
            arr = FancyArrowPatch((x + w + 0.02, y0 + h / 2),
                                  (i + 1 + (1 - w) / 2 - 0.02, y0 + h / 2),
                                  arrowstyle="-|>", mutation_scale=13,
                                  color=MUTED, lw=1.5)
            ax.add_patch(arr)
    ax.text(len(blocks) / 2, 0.90,
            "SITSTransformer  ·  self-attention over the time steps",
            ha="center", fontsize=13, color=INK, weight="bold")
    ax.text(len(blocks) / 2, 0.06,
            f"input-agnostic: the SAME model reads NDVI (T={T}, C=1) or "
            f"AlphaEarth embeddings (T={len(AE_YEARS)}, C={AE_DIMS})",
            ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_06_transformer.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_06_transformer.png")


# =============================================================================
# 7/8. ALPHAEARTH -- embeddings, model comparison and PCA
# =============================================================================
def load_alphaearth(df):
    """Return (ae_df, is_synth). Synthetic fallback mirrors the notebook's."""
    if AE_CSV.exists():
        ae = pd.read_csv(AE_CSV)
        if len(ae) == len(df):
            return ae, False
        print(f"  (AlphaEarth CSV has {len(ae)} rows, expected {len(df)} -- "
              "using synthetic fallback)")
    rng = np.random.default_rng(SEED)
    labels = df["label"].tolist()
    centre = {"forest": rng.normal(0, 0.3, AE_DIMS),
              "old_clearing": rng.normal(0, 0.3, AE_DIMS)}
    frames = [pd.DataFrame({"label": labels})]
    for year in range(2018, 2025):
        cols = [f"emb_{year}_{b:02d}" for b in range(AE_DIMS)]
        mat = np.empty((len(labels), AE_DIMS), dtype=np.float32)
        for i, lab in enumerate(labels):
            key = ("old_clearing"
                   if lab == "old_clearing" or (lab == "deforested" and year >= 2022)
                   else "forest")
            mat[i] = centre[key] + rng.normal(0, 0.15, AE_DIMS)
        frames.append(pd.DataFrame(mat, columns=cols))
    return pd.concat(frames, axis=1), True


def synth_watermark(fig):
    fig.text(0.5, 0.5, "SYNTHETIC PLACEHOLDER\nrun build_alphaearth.py, then re-run",
             ha="center", va="center", fontsize=22, color="#e34948",
             alpha=0.25, rotation=20, weight="bold", zorder=10)


def metrics_row(yte, pred, order):
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return {
        "accuracy": accuracy_score(yte, pred),
        "macro-F1": f1_score(yte, pred, average="macro"),
        "deforested\nrecall": recall_score(
            yte, pred, labels=list(range(len(order))),
            average=None, zero_division=0)[order.index("deforested")],
    }


def fig_model_comparison(results, ae_is_synth):
    """Grouped bars: one group per metric, one bar per model."""
    MODEL_COLOR = {
        "Random Forest": MUTED,
        "1D-CNN": SEQ_BLUE[3],
        "Transformer (NDVI)": SEQ_BLUE[5],
        "Transformer (AlphaEarth)": "#008300",
    }
    metrics = list(next(iter(results.values())).keys())
    models = list(results.keys())
    xs = np.arange(len(metrics))
    bw = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    clean(ax)
    for mi, model in enumerate(models):
        vals = [results[model][m] for m in metrics]
        pos = xs + (mi - (len(models) - 1) / 2) * bw
        bars = ax.bar(pos, vals, width=bw * 0.92,
                      color=MODEL_COLOR.get(model, SEQ_BLUE[2]), label=model)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score (test set)")
    ax.set_title("Four models, three honest numbers", fontsize=12.5, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9)
    if ae_is_synth:
        synth_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig_07_model_comparison.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_07_model_comparison.png")


def fig_alphaearth_pca(ae_df, order, is_synth, year=2022):
    """2-D PCA of one year's 64-d embeddings, coloured by class."""
    cols = [f"emb_{year}_{b:02d}" for b in range(AE_DIMS)]
    valid = ae_df[cols].notna().all(axis=1)
    X = ae_df.loc[valid, cols].to_numpy("float32")
    labels = ae_df.loc[valid, "label"].to_numpy()
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    clean(ax)
    for cls in order:
        st = CLASS_STYLE[cls]
        m = labels == cls
        ax.scatter(pcs[m, 0], pcs[m, 1], s=30, marker=st["marker"],
                   c=st["color"], edgecolors=SURFACE, linewidths=0.5,
                   alpha=0.85, label=st["label"])
    ax.set_xlabel("principal component 1"); ax.set_ylabel("principal component 2")
    ax.set_title(f"AlphaEarth embeddings ({year}), projected to 2-D\n"
                 "the foundation model separates the classes before we train anything",
                 fontsize=12, pad=10)
    ax.legend(loc="best", frameon=True, fontsize=9, framealpha=0.95,
              edgecolor=GRID)
    if is_synth:
        synth_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig_08_alphaearth_pca.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("  wrote fig_08_alphaearth_pca.png")


def main():
    if not DATA_CSV.exists():
        raise SystemExit(f"Dataset not found: {DATA_CSV}. Run build_dataset.py first.")
    df, ndvi_cols, order = load()
    print(f"Loaded {len(df)} samples, {len(ndvi_cols)} timesteps, classes={order}")

    print("1) data-input figures ...")
    fig_data_input(df, ndvi_cols, order)
    anim_ndvi_reveal(df, ndvi_cols, order)
    print("2) labels map ...")
    fig_labels(df, ndvi_cols, order)
    print("3) model diagram ...")
    fig_model(len(ndvi_cols), len(order))
    print("4) training the CNN (real pipeline) ...")
    hist, yte, pred, _ = train_cnn(df, ndvi_cols, order)
    anim_training(hist, len(order))
    print("5) inference figures ...")
    fig_confusion(yte, pred, order)
    fig_inference_map(df, pred, yte, order)

    print("6) transformer diagram ...")
    fig_transformer(len(ndvi_cols), len(order))

    print("7) training the other models for the comparison ...")
    from sklearn.ensemble import RandomForestClassifier

    cls2idx = {c: i for i, c in enumerate(order)}
    X = df[ndvi_cols].to_numpy("float32")
    y = df["label"].map(cls2idx).to_numpy()
    test_mask = split_mask(df)

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X[~test_mask], y[~test_mask])
    rf_pred = rf.predict(X[test_mask])

    _, _, tr_pred = train_torch_model(
        build_transformer(len(order), in_dim=1, seq_len=X.shape[1]),
        X[:, :, None], y, test_mask)

    ae_df, ae_is_synth = load_alphaearth(df)
    if ae_is_synth:
        print("  (AlphaEarth CSV not found: figs 07/08 use watermarked synthetic data)")
    emb_cols = [f"emb_{yr}_{b:02d}" for yr in AE_YEARS for b in range(AE_DIMS)]
    valid = ae_df[emb_cols].notna().all(axis=1).to_numpy()
    X_emb = (ae_df.loc[valid, emb_cols].to_numpy("float32")
             .reshape(valid.sum(), len(AE_YEARS), AE_DIMS))
    _, yte_e, ae_pred = train_torch_model(
        build_transformer(len(order), in_dim=AE_DIMS, seq_len=len(AE_YEARS)),
        X_emb, y[valid], test_mask[valid])

    results = {
        "Random Forest": metrics_row(y[test_mask], rf_pred, order),
        "1D-CNN": metrics_row(yte, pred, order),
        "Transformer (NDVI)": metrics_row(y[test_mask], tr_pred, order),
        "Transformer (AlphaEarth)": metrics_row(yte_e, ae_pred, order),
    }
    fig_model_comparison(results, ae_is_synth)

    print("8) AlphaEarth PCA ...")
    fig_alphaearth_pca(ae_df, order, ae_is_synth)
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
