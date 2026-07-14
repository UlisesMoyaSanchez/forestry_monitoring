#!/usr/bin/env python3
"""Smoke-test for the CCAI deforestation-SITS tutorial.

Checks that everything the participants run actually works end-to-end:
  1. all required libraries import;
  2. the dataset loads (falls back to a small synthetic SITS set if the real
     CSV has not been built yet, so the modelling code can always be tested);
  3. the scikit-learn baseline trains and beats chance;
  4. the 1D-CNN trains for a few epochs and beats chance;
  5. CodeCarbon can instrument a block without crashing;
  6. the time series transformer trains on the NDVI series and beats chance;
  7. the AlphaEarth embeddings load (real CSV if built, else the synthetic
     fallback) and the same transformer does a forward/backward pass on them.

Exit code 0 => "all runs great".  Run:  python check_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_CSV = REPO / "data" / "amazon_sits_samples.csv"
AE_CSV = REPO / "data" / "amazon_alphaearth_samples.csv"
AE_YEARS = list(range(2018, 2023))  # notebook default: up to the target year
AE_DIMS = 64

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name, fn):
    try:
        msg = fn() or ""
        results.append((name, True, msg))
        print(f"[{PASS}] {name}  {msg}")
        return True
    except Exception as e:  # noqa: BLE001
        results.append((name, False, repr(e)))
        print(f"[{FAIL}] {name}  {e!r}")
        return False


# ---------------------------------------------------------------------------
# 1. imports
# ---------------------------------------------------------------------------
def _imports():
    import matplotlib  # noqa: F401
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import sklearn  # noqa: F401
    import torch

    return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"


# ---------------------------------------------------------------------------
# 2. dataset (real if present, else synthetic fallback)
# ---------------------------------------------------------------------------
def make_synthetic(n_per_class=200, n_t=24, seed=0):
    """Realistic-ish NDVI trajectories so the pipeline is testable offline."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_t)
    rows, labels = [], []

    def noisy(base):
        return np.clip(base + rng.normal(0, 0.03, n_t), -1, 1)

    for _ in range(n_per_class):  # forest: high, flat, mild seasonality
        rows.append(noisy(0.85 + 0.03 * np.sin(2 * np.pi * t)))
        labels.append("forest")
    for _ in range(n_per_class):  # deforested this year: high then sharp drop
        drop = rng.integers(int(0.3 * n_t), int(0.8 * n_t))
        base = np.where(np.arange(n_t) < drop, 0.85, 0.25)
        rows.append(noisy(base))
        labels.append("deforested")
    for _ in range(n_per_class):  # old clearing / pasture: low, seasonal
        rows.append(noisy(0.4 + 0.15 * np.sin(2 * np.pi * t)))
        labels.append("old_clearing")

    cols = [f"ndvi_{i:02d}" for i in range(n_t)]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "label", labels)
    df.insert(0, "lat", rng.uniform(-5.1, -4.9, len(df)))
    df.insert(0, "lon", rng.uniform(-54.9, -54.7, len(df)))
    return df


def load_dataset():
    import pandas as pd

    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        src = f"real dataset ({DATA_CSV.name})"
    else:
        df = make_synthetic()
        src = "SYNTHETIC fallback (real CSV not built yet)"
    ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
    assert ndvi_cols, "no ndvi_* columns found"
    assert df["label"].nunique() >= 2, "need >= 2 classes"
    globals()["_DF"] = df
    globals()["_NDVI_COLS"] = ndvi_cols
    return f"{src}: {len(df)} samples, {len(ndvi_cols)} timesteps, {df['label'].nunique()} classes"


# ---------------------------------------------------------------------------
# helpers shared by the model checks
# ---------------------------------------------------------------------------
def _xy():
    import numpy as np

    df, cols = globals()["_DF"], globals()["_NDVI_COLS"]
    X = df[cols].to_numpy("float32")
    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return X, y, classes


def _split(X, y, seed=0):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


# ---------------------------------------------------------------------------
# 3. RandomForest baseline
# ---------------------------------------------------------------------------
def _rf_baseline():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y, classes = _xy()
    Xtr, Xte, ytr, yte = _split(X, y)
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    chance = 1.0 / len(classes)
    assert acc > chance + 0.1, f"RF acc {acc:.2f} not above chance {chance:.2f}"
    return f"test acc={acc:.3f} (chance={chance:.3f})"


# ---------------------------------------------------------------------------
# 4. 1D CNN
# ---------------------------------------------------------------------------
def build_cnn(n_classes, n_channels=1):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(n_channels, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(32, n_classes),
    )


def _cnn_train():
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score

    X, y, classes = _xy()
    Xtr, Xte, ytr, yte = _split(X, y)
    to_t = lambda a: torch.tensor(a).unsqueeze(1)  # (N, 1, T)
    Xtr_t, Xte_t = to_t(Xtr), to_t(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    torch.manual_seed(0)
    model = build_cnn(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    acc = accuracy_score(yte, pred)
    chance = 1.0 / len(classes)
    assert acc > chance + 0.1, f"CNN acc {acc:.2f} not above chance {chance:.2f}"
    return f"test acc={acc:.3f} after 150 epochs (chance={chance:.3f})"


# ---------------------------------------------------------------------------
# 5/6. time series transformer (same architecture as the notebook)
# ---------------------------------------------------------------------------
def build_transformer(n_classes, in_dim=1, seq_len=24,
                      d_model=32, n_heads=2, n_layers=1, dropout=0.2):
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

        def forward(self, x):  # (B, T, C)
            h = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
            return self.head(self.encoder(h).mean(dim=1))

    return SITSTransformer()


def _transformer_train():
    import torch
    from sklearn.metrics import accuracy_score

    X, y, classes = _xy()
    Xtr, Xte, ytr, yte = _split(X, y)
    to_t = lambda a: torch.tensor(a).unsqueeze(-1)  # (N, T, 1)
    Xtr_t, Xte_t = to_t(Xtr), to_t(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    torch.manual_seed(0)
    model = build_transformer(len(classes), in_dim=1, seq_len=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    acc = accuracy_score(yte, pred)
    chance = 1.0 / len(classes)
    assert acc > chance + 0.1, f"transformer acc {acc:.2f} not above chance {chance:.2f}"
    return f"test acc={acc:.3f} after 150 epochs (chance={chance:.3f})"


# ---------------------------------------------------------------------------
# 7. AlphaEarth embeddings (real CSV if built, else synthetic fallback)
# ---------------------------------------------------------------------------
def make_synthetic_alphaearth(labels, years=AE_YEARS, n_dims=AE_DIMS, seed=0):
    """Per-class Gaussian clusters in 64-d; 'deforested' jumps centres in 2022."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    centre = {"forest": rng.normal(0, 0.3, n_dims),
              "old_clearing": rng.normal(0, 0.3, n_dims)}
    frames = [pd.DataFrame({"label": labels})]
    for year in years:
        cols = [f"emb_{year}_{b:02d}" for b in range(n_dims)]
        mat = np.empty((len(labels), n_dims), dtype=np.float32)
        for i, lab in enumerate(labels):
            key = ("old_clearing"
                   if lab == "old_clearing" or (lab == "deforested" and year >= 2022)
                   else "forest")
            mat[i] = centre[key] + rng.normal(0, 0.15, n_dims)
        frames.append(pd.DataFrame(mat, columns=cols))
    return pd.concat(frames, axis=1)


def _alphaearth():
    import numpy as np
    import pandas as pd
    import torch

    df = globals()["_DF"]
    if AE_CSV.exists():
        ae = pd.read_csv(AE_CSV)
        assert len(ae) == len(df), "embedding CSV not row-aligned with NDVI CSV"
        src = f"real embeddings ({AE_CSV.name})"
    else:
        ae = make_synthetic_alphaearth(df["label"].tolist())
        src = "SYNTHETIC embeddings (real CSV not built yet)"

    emb_cols = [f"emb_{y}_{b:02d}" for y in AE_YEARS for b in range(AE_DIMS)]
    valid = ae[emb_cols].notna().all(axis=1).to_numpy()
    X_emb = (ae.loc[valid, emb_cols].to_numpy("float32")
             .reshape(valid.sum(), len(AE_YEARS), AE_DIMS))
    _, y, classes = _xy()
    y_emb = torch.tensor(y[valid], dtype=torch.long)

    torch.manual_seed(0)
    model = build_transformer(len(classes), in_dim=AE_DIMS, seq_len=len(AE_YEARS))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss = torch.nn.CrossEntropyLoss()(model(torch.tensor(X_emb)), y_emb)
    loss.backward()
    opt.step()
    assert np.isfinite(loss.item()), "non-finite loss on embeddings"
    return f"{src}: X={tuple(X_emb.shape)}, fwd/bwd OK (loss={loss.item():.3f})"


# ---------------------------------------------------------------------------
# 5. CodeCarbon
# ---------------------------------------------------------------------------
def _codecarbon():
    from codecarbon import EmissionsTracker

    tr = EmissionsTracker(save_to_file=False, logging_logger=None, log_level="error")
    tr.start()
    _ = sum(i * i for i in range(100000))
    emissions = tr.stop()
    return f"tracked {emissions:.2e} kg CO2e for a trivial block"


# ---------------------------------------------------------------------------
def main():
    print("=" * 66)
    print("CCAI deforestation-SITS tutorial :: pipeline smoke test")
    print("=" * 66)
    ok = True
    ok &= check("1. imports", _imports)
    ok &= check("2. dataset load", load_dataset)
    ok &= check("3. RandomForest baseline", _rf_baseline)
    ok &= check("4. 1D-CNN trains", _cnn_train)
    ok &= check("5. CodeCarbon instrumentation", _codecarbon)
    ok &= check("6. transformer trains", _transformer_train)
    ok &= check("7. AlphaEarth embeddings + transformer", _alphaearth)
    print("-" * 66)
    n_pass = sum(1 for _, p, _ in results if p)
    print(f"{n_pass}/{len(results)} checks passed")
    if ok:
        print("\n\033[92mAll runs great.\033[0m")
    else:
        print("\n\033[91mSome checks failed (see above).\033[0m")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
