"""
M3 Model Comparison & Ablation Study
====================================
Benchmarks the SAFE masked autoencoder against alternative anomaly detectors
under IDENTICAL conditions (same WOA features, same normal-only training data,
same RobustScaler, same stratified test sample, same validation threshold rule).

Models compared
  Classical ML : Isolation Forest, One-Class SVM, LOF, PCA reconstruction
  Deep AE      : AE (no masking), DAE (Gaussian noise), VAE,
                 SAFE-MAE (ours), SAFE-MAE + multi-mask scoring
  Transformer  : Transformer masked autoencoder (FT-Transformer-style tokens)
  Reference    : Random Forest (SUPERVISED; off by default, --with-supervised)

Ablations
  * Mask-ratio sweep for SAFE-MAE (0.0 = plain AE ... 0.7)
  * WOA features vs. all features

Outputs (results/comparison/)
  raw_results.csv, summary_results.csv, mask_ablation.csv, woa_ablation.csv,
  summary.md, fig1_metrics_bar.png, fig2_curves_<dataset>.png,
  fig3_efficiency.png, fig4_mask_ablation.png, fig5_woa_ablation.png,
  fig6_score_distribution.png

Usage (from the project root, same place you run the other modules):
  python -m src.evaluation.compare_models            # full run (3 seeds)
  python -m src.evaluation.compare_models --quick    # 1 seed, 10 epochs (smoke test)
  python -m src.evaluation.compare_models --datasets MedSec-25 --seeds 42 7
  python -m src.evaluation.compare_models --use-best  # SAFE uses tuned config from tune_safe.py
Place this file next to evaluator.py (src/evaluation/). It reuses the data
loaders and the MaskedAutoencoder class from your safe.py, so "SAFE-MAE (ours)"
is exactly your model.
"""

import os
import io
import gc
import sys
import glob
import json
import time
import argparse
import contextlib
import importlib.util
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ============================================================
# PATHS & IMPORT OF YOUR safe.py
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "comparison")


def import_safe_module():
    candidates = [
        p for p in glob.glob(os.path.join(PROJECT_ROOT, "src", "**", "safe.py"), recursive=True)
        if os.path.abspath(p) != os.path.abspath(__file__)
    ]
    if not candidates:
        raise FileNotFoundError("Could not find safe.py under src/. Place this script in src/evaluation/.")
    spec = importlib.util.spec_from_file_location("safe_module", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"[compare] Using SAFE implementation from: {candidates[0]}")
    return module


safe = import_safe_module()
DEVICE = torch.device("cpu")


# ============================================================
# SETTINGS
# ============================================================

DATASETS = ["IoMT-TrafficData", "MedSec-25"]

# Columns that encode the label itself. In IoMT-TrafficData, 'traffic' is the
# attack-type name (apachekiller, arpspoofing, ..., normal) label-encoded as a
# feature. Keeping it lets every model "detect" attacks by reading the answer.
LEAKY_COLUMNS = ["traffic", "Label", "label", "is_attack", "attack_type", "Attack Type"]

MASK_RATIO = 0.20                     # same as safe.py
MASK_ABLATION_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
N_SCORE_MASKS = 8                     # masks averaged in multi-mask scoring

OUR_MODEL = "SAFE-MAE (ours)"


# ============================================================
# DATA PREPARATION (identical pipeline to safe.py)
# ============================================================

def prepare_dataset(name, use_woa=True, drop_leaky=True):
    dataset_dir = os.path.join(safe.PROCESSED_DIR, name)
    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    label = safe.detect_label_column(train_path)
    normal_label = safe.determine_normal_label(name, train_path, label)

    if use_woa:
        cols = safe.resolve_feature_columns(train_path, safe.load_selected_features(name))
    else:
        header = pd.read_csv(train_path, nrows=0).columns
        cols = [c for c in header if c != label]

    leaked = [c for c in cols if c in LEAKY_COLUMNS]
    if leaked:
        print(f"\n[compare] WARNING: label-leaking column(s) in feature set: {leaked}")
        if drop_leaky:
            cols = [c for c in cols if c not in LEAKY_COLUMNS]
            print(f"[compare] Dropped them -> {len(cols)} features remain.")
        else:
            print("[compare] Kept them (--keep-leaky). Results will be optimistic.")

    X_normal = safe.load_normal_training_data(
        train_path, cols, label, normal_label, safe.MAX_NORMAL_TRAIN_SAMPLES)
    n_val = max(100, int(len(X_normal) * safe.VALIDATION_FRACTION))
    X_train, X_val = X_normal[:-n_val], X_normal[-n_val:]

    X_attack_val = safe.load_attack_validation_data(
        train_path, cols, label, normal_label, safe.MAX_ATTACK_VALIDATION_SAMPLES)

    X_test, y_test = safe.load_test_data(
        test_path, cols, label, normal_label, safe.MAX_TEST_SAMPLES)

    scaler = RobustScaler().fit(X_train)

    def transform(X):
        return np.clip(scaler.transform(X), -10.0, 10.0).astype(np.float32)

    return {
        "name": name,
        "features": cols,
        "Xtr": transform(X_train),
        "Xval": transform(X_val),
        "Xatk": transform(X_attack_val) if X_attack_val is not None and len(X_attack_val) else None,
        "Xte": transform(X_test),
        "yte": (y_test != normal_label).astype(int),
    }


def pick_threshold(val_normal_scores, val_attack_scores):
    """Same rule as safe.py (Youden's J or max-F1 depending on Cohen's d)."""
    if val_attack_scores is None or len(val_attack_scores) == 0:
        return float(np.percentile(val_normal_scores, safe.THRESHOLD_PERCENTILE))
    with contextlib.redirect_stdout(io.StringIO()):
        return float(safe.calculate_threshold_validation(
            np.asarray(val_normal_scores), np.asarray(val_attack_scores)))


# ============================================================
# EXTRA NEURAL ARCHITECTURES
# ============================================================

class ConfigurableAE(nn.Module):
    """Same layout as safe.MaskedAutoencoder; with hidden=(128, 64), latent=32,
    dropout=0.1 it is the identical network. Used for hyperparameter tuning."""

    def __init__(self, input_dim, hidden=(128, 64), latent=32, dropout=0.1):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, latent), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


DEFAULT_SAFE_CONFIG = {"epochs": 30, "mask_ratio": 0.2, "lr": 5e-4, "batch_size": 128,
                       "latent": 32, "dropout": 0.1, "loss": "huber", "scorer": "recon"}


def load_best_config(dataset):
    """Use exactly the configuration safe.py uses (FINAL_CONFIGS / tuned file)."""
    cfg = {k: v for k, v in safe.load_safe_config(dataset).items() if k in DEFAULT_SAFE_CONFIG}
    if isinstance(cfg["mask_ratio"], (list, tuple)):
        # safe.py selects the mask ratio on validation; reuse its choice
        meta_path = os.path.join(PROJECT_ROOT, "results", "safe", dataset, "safe_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{dataset}: safe.py searches over mask ratios {cfg['mask_ratio']}. "
                f"Run src/safe/safe.py first so its validation-selected mask ratio can be reused.")
        with open(meta_path, encoding="utf-8") as f:
            cfg["mask_ratio"] = float(json.load(f)["mask_ratio"])
        print(f"[compare] {dataset}: mask ratio {cfg['mask_ratio']} (selected on validation by safe.py)")
    print(f"[compare] Using SAFE config for {dataset} from safe.py: {cfg}")
    return cfg


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec(z), mu, logvar


class TabTransformerMAE(nn.Module):
    """
    Transformer masked autoencoder for tabular flows (FT-Transformer-style).
    Every feature becomes a token: x_j * w_j + b_j; masked tokens are replaced by
    a learned [MASK] token; a per-feature positional embedding is added; a small
    Transformer encoder models feature-feature interactions; a linear head
    reconstructs each scalar.
    """
    uses_mask_token = True

    def __init__(self, input_dim, d_model=32, nhead=4, layers=2, ff=64, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim, d_model) * 0.1)
        self.b = nn.Parameter(torch.zeros(input_dim, d_model))
        self.pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        tok = x.unsqueeze(-1) * self.w + self.b
        if mask is not None:
            tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        h = self.encoder(tok + self.pos)
        return self.head(h).squeeze(-1)


# ============================================================
# DETECTOR WRAPPERS  (score: higher = more anomalous)
# ============================================================

class SklearnDetector:
    supervised = False

    def __init__(self, factory, max_train=None):
        self.factory, self.max_train = factory, max_train

    def fit(self, data, seed):
        X = data["Xtr"]
        if self.max_train and len(X) > self.max_train:
            X = X[np.random.default_rng(seed).choice(len(X), self.max_train, replace=False)]
        self.model = self.factory(seed).fit(X)

    def score(self, X):
        return -self.model.decision_function(X)

    def n_params(self):
        return np.nan


class PCADetector(SklearnDetector):
    def __init__(self):
        pass

    def fit(self, data, seed):
        self.model = PCA(n_components=0.95, random_state=seed).fit(data["Xtr"])

    def score(self, X):
        R = self.model.inverse_transform(self.model.transform(X))
        return ((R - X) ** 2).mean(axis=1)

    def n_params(self):
        return int(self.model.components_.size)


class SupervisedRF(SklearnDetector):
    supervised = True

    def __init__(self):
        pass

    def fit(self, data, seed):
        Xa, Xn = data["Xatk"], data["Xtr"]
        rng = np.random.default_rng(seed)
        Xn = Xn[rng.choice(len(Xn), min(len(Xn), 5 * len(Xa)), replace=False)]
        X = np.vstack([Xn, Xa])
        y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
        self.model = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed, class_weight="balanced").fit(X, y)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class TorchDetector:
    supervised = False

    def __init__(self, build, corruption="none", mask_ratio=0.0, noise_std=0.1,
                 epochs=30, batch_size=128, lr=5e-4, kl_beta=0.0, scoring="full", loss="huber",
                 scorer=None, lof_k=20):
        self.build, self.corruption = build, corruption
        self.mask_ratio, self.noise_std = mask_ratio, noise_std
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.kl_beta, self.scoring, self.loss = kl_beta, scoring, loss
        # scorer: None (plain reconstruction error, baselines) or one of
        # "recon" / "latent_lof" / "hybrid" / "auto" -> identical code to safe.py
        self.scorer, self.lof_k = scorer, lof_k
        self.scorer_obj, self.chosen = None, None

    @staticmethod
    def _recon(out):
        return out[0] if isinstance(out, tuple) else out

    def _forward(self, x, mask=None):
        if mask is None:
            return self.net(x)
        if getattr(self.net, "uses_mask_token", False):
            return self.net(x, mask)
        x = x.clone()
        x[mask] = 0.0          # same corruption as safe.py (0 = median after RobustScaler)
        return self.net(x)

    def fit(self, data, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = torch.from_numpy(data["Xtr"])
        self.net = self.build(X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss() if self.loss == "mse" else nn.SmoothL1Loss()
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True,
                            drop_last=True, generator=torch.Generator().manual_seed(seed))
        self.net.train()
        for _ in range(self.epochs):
            for (b,) in loader:
                if self.corruption == "mask" and self.mask_ratio > 0:
                    out = self._forward(b, torch.rand_like(b) < self.mask_ratio)
                elif self.corruption == "gaussian":
                    out = self.net(b + self.noise_std * torch.randn_like(b))
                else:
                    out = self.net(b)
                loss = loss_fn(self._recon(out), b)
                if isinstance(out, tuple):
                    _, mu, logvar = out
                    loss = loss + self.kl_beta * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                opt.zero_grad()
                loss.backward()
                opt.step()

        if self.scorer is not None and self.scoring == "full":
            self.net.eval()
            if self.scorer == "auto":
                with contextlib.redirect_stdout(io.StringIO()):
                    self.scorer_obj, _ = safe.select_scorer(
                        self.net, data["Xtr"], data["Xval"], data["Xatk"], "auto")
            else:
                self.scorer_obj = safe.SAFEScorer(self.net, self.scorer, self.lof_k).fit(data["Xtr"])
            self.chosen = (self.scorer_obj.kind, self.scorer_obj.n_neighbors)

    def score(self, X):
        if self.scorer_obj is not None:
            return self.scorer_obj.score(X)
        self.net.eval()
        scores = []
        gen = torch.Generator().manual_seed(0)
        ratio = self.mask_ratio if self.mask_ratio > 0 else MASK_RATIO
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                b = torch.from_numpy(X[i:i + 2048])
                if self.scoring == "multimask":
                    acc = torch.zeros(len(b))
                    for _ in range(N_SCORE_MASKS):
                        m = torch.rand(b.shape, generator=gen) < ratio
                        r = self._recon(self._forward(b, m))
                        acc += ((r - b) ** 2 * m).sum(1) / m.sum(1).clamp(min=1)
                    scores.append((acc / N_SCORE_MASKS).numpy())
                else:
                    r = self._recon(self._forward(b))
                    scores.append(((r - b) ** 2).mean(1).numpy())
        return np.concatenate(scores)

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


def make_safe(cfg, scoring="full", mask_ratio=None, scorer=None):
    r = cfg["mask_ratio"] if mask_ratio is None else mask_ratio
    return TorchDetector(lambda d: ConfigurableAE(d, latent=cfg["latent"], dropout=cfg["dropout"]),
                         "mask" if r > 0 else "none", r, epochs=cfg["epochs"],
                         batch_size=cfg["batch_size"], lr=cfg["lr"], loss=cfg["loss"], scoring=scoring,
                         scorer=scorer or cfg.get("scorer", "recon"), lof_k=cfg.get("lof_k", 20))


def build_models(epochs, safe_cfg=None, with_supervised=False):
    MAE = safe.MaskedAutoencoder
    safe_cfg = safe_cfg or {**DEFAULT_SAFE_CONFIG, "epochs": epochs}
    models = [
        ("Isolation Forest", "Classical ML",
         lambda: SklearnDetector(lambda s: IsolationForest(n_estimators=200, random_state=s, n_jobs=-1))),
        ("One-Class SVM", "Classical ML",
         lambda: SklearnDetector(lambda s: OneClassSVM(nu=0.05, gamma="scale"), max_train=10000)),
        ("LOF", "Classical ML",
         lambda: SklearnDetector(lambda s: LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
                                 max_train=20000)),
        ("PCA reconstruction", "Classical ML", lambda: PCADetector()),
        ("AE (no masking)", "Deep AE", lambda: TorchDetector(MAE, "none", epochs=epochs)),
        ("DAE (Gaussian noise)", "Deep AE", lambda: TorchDetector(MAE, "gaussian", epochs=epochs)),
        ("VAE", "Deep AE", lambda: TorchDetector(VAE, "none", epochs=epochs, kl_beta=0.1)),
        ("Transformer MAE", "Transformer",
         lambda: TorchDetector(TabTransformerMAE, "mask", MASK_RATIO, epochs=epochs, batch_size=256)),
        (OUR_MODEL, "Deep AE (ours)", lambda: make_safe(safe_cfg)),
        ("SAFE recon. score only", "Deep AE (ours)", lambda: make_safe(safe_cfg, scorer="recon")),
        ("SAFE encoder + LOF", "Deep AE (ours)", lambda: make_safe(safe_cfg, scorer="latent_lof")),
        ("SAFE-MAE + multi-mask score", "Deep AE (ours)", lambda: make_safe(safe_cfg, scoring="multimask")),
    ]
    if with_supervised:
        models.append(("Random Forest (supervised ref.)", "Supervised reference", lambda: SupervisedRF()))
    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate(detector, data, seed):
    t0 = time.perf_counter()
    detector.fit(data, seed)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_test = detector.score(data["Xte"])
    latency_us = (time.perf_counter() - t0) / len(s_test) * 1e6

    if detector.supervised:
        thr = 0.5
    else:
        s_atk = detector.score(data["Xatk"]) if data["Xatk"] is not None else None
        thr = pick_threshold(detector.score(data["Xval"]), s_atk)

    y = data["yte"]
    pred = (s_test > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    extra = safe.imbalance_aware_metrics(int(tp), int(fp), int(tn), int(fn))
    metrics = {
        "roc_auc": roc_auc_score(y, s_test),
        "pr_auc": average_precision_score(y, s_test),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "fpr": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / len(y),
        "specificity": extra["specificity"],
        "balanced_acc": extra["balanced_accuracy"],
        "mcc": extra["mcc"],
        "threshold": thr,
        "scorer": "" if getattr(detector, "chosen", None) is None
                  else f"{detector.chosen[0]}" + ("" if detector.chosen[0] == "recon" else f"(k={detector.chosen[1]})"),
        "train_s": train_s,
        "latency_us": latency_us,
        "n_params": detector.n_params(),
    }
    return metrics, s_test


# ============================================================
# PLOTS
# ============================================================

def _color(model):
    if model.startswith("SAFE"):
        return "#d62728"
    if "supervised" in model:
        return "#7f7f7f"
    return None


def plot_metric_bars(agg, datasets, order):
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("balanced_acc", "Balanced acc.")]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4.8 * len(datasets)), squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        sub = agg[agg.dataset == ds].set_index("model")
        sub = sub.loc[[m for m in order if m in sub.index]]
        x, w = np.arange(len(sub)), 0.27
        for i, (m, lbl) in enumerate(metrics):
            ax.bar(x + (i - 1) * w, sub[f"{m}_mean"], w, yerr=sub[f"{m}_std"].fillna(0),
                   capsize=2, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=28, ha="right")
        for t in ax.get_xticklabels():
            if t.get_text().startswith("SAFE"):
                t.set_fontweight("bold")
                t.set_color("#d62728")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"{ds}: detection performance (mean ± std over seeds)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_metrics_bar.png"), dpi=200)
    plt.close(fig)


def plot_curves(curves, ds, order):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for model in order:
        if model not in curves:
            continue
        y, s = curves[model]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        style = dict(color=_color(model), lw=2.6 if model == OUR_MODEL else 1.3,
                     ls="--" if "supervised" in model else "-")
        a1.plot(fpr, tpr, label=f"{model} ({roc_auc_score(y, s):.3f})", **style)
        a2.plot(rec, prec, label=f"{model} ({average_precision_score(y, s):.3f})", **style)
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {ds}")
    a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall - {ds}")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="lower right" if a is a1 else "lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig2_curves_{ds}.png"), dpi=200)
    plt.close(fig)


def plot_efficiency(agg, datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = agg[agg.dataset == ds]
        for _, r in sub.iterrows():
            ax.scatter(r.latency_us_mean, r.roc_auc_mean, s=90,
                       color=_color(r.model) or "#1f77b4", edgecolor="k", zorder=3)
            ax.annotate(r.model, (r.latency_us_mean, r.roc_auc_mean), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set(xlabel="Inference latency per flow (µs, log scale)", ylabel="ROC-AUC",
               title=f"{ds}: accuracy vs. cost")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_efficiency.png"), dpi=200)
    plt.close(fig)


def plot_mask_ablation(df_mask):
    agg = df_mask.groupby(["dataset", "mask_ratio"])[["roc_auc", "f1"]].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, title in zip(axes, ["roc_auc", "f1"], ["ROC-AUC", "F1"]):
        for ds, g in agg.groupby("dataset"):
            m, s = g[(metric, "mean")].values, g[(metric, "std")].fillna(0).values
            ax.plot(g["mask_ratio"], m, marker="o", label=ds)
            ax.fill_between(g["mask_ratio"], m - s, m + s, alpha=0.2)
        ax.set(xlabel="Mask ratio during training (0 = plain AE)", ylabel=title,
               title=f"SAFE-MAE mask-ratio ablation: {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_mask_ablation.png"), dpi=200)
    plt.close(fig)


def plot_woa_ablation(df_woa):
    agg = df_woa.groupby(["dataset", "setting"], sort=False)[["roc_auc", "f1", "latency_us", "train_s"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title in zip(axes, ["roc_auc", "f1", "train_s"],
                                 ["ROC-AUC", "F1", "Training time (s)"]):
        datasets = agg.dataset.unique()
        x, w = np.arange(len(datasets)), 0.38
        for i, key in enumerate(["All features", "WOA features"]):
            vals = [agg[(agg.dataset == d) & agg.setting.str.startswith(key)][metric].values[0] for d in datasets]
            labels = [agg[(agg.dataset == d) & agg.setting.str.startswith(key)].setting.values[0] for d in datasets]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=key)
            for b, l in zip(bars, labels):
                ax.annotate(l.split("(")[-1].rstrip(")"), (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Effect of WOA feature selection on SAFE-MAE")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_woa_ablation.png"), dpi=200)
    plt.close(fig)


def plot_score_distribution(dists):
    fig, axes = plt.subplots(1, len(dists), figsize=(7 * len(dists), 4.5), squeeze=False)
    for ax, (ds, (y, s, thr)) in zip(axes[0], dists.items()):
        ls = np.log10(np.maximum(s, 1e-9))
        bins = np.linspace(ls.min(), ls.max(), 80)
        ax.hist(ls[y == 0], bins=bins, alpha=0.6, density=True, label="Normal")
        ax.hist(ls[y == 1], bins=bins, alpha=0.6, density=True, label="Attack")
        ax.axvline(np.log10(max(thr, 1e-9)), color="k", ls="--", label="Threshold")
        ax.set(xlabel="log10(reconstruction error)", ylabel="Density",
               title=f"{ds}: SAFE-MAE anomaly-score separation")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_score_distribution.png"), dpi=200)
    plt.close(fig)


def write_summary_md(agg, df_mask, df_woa, datasets, order, info=None):
    info = info or {}
    lines = ["# M3 Model Comparison Summary", "",
             f"Thresholds: rule from safe.py with validation FPR capped at {safe.MAX_VALIDATION_FPR:.0%}.", ""]
    for ds in datasets:
        sub = agg[agg.dataset == ds].set_index("model")
        lines += [f"## {ds}", ""]
        if ds in info and not np.isnan(info[ds]["attack_share"]):
            p = info[ds]["attack_share"]
            lines += [f"Attack share in test set: {p:.2%}. Trivial 'flag everything' baseline: "
                      f"accuracy {p:.4f}, F1 {2 * p / (1 + p):.4f}, balanced accuracy 0.5000, MCC 0.",
                      f"SAFE scorer selected on validation: **{info[ds].get('scorer', 'n/a')}**.", ""]
        lines += ["| Model | Family | ROC-AUC | PR-AUC | Bal. acc. | MCC | F1 | Precision | Recall | FPR | Train (s) | Latency (µs/flow) | Params |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in order:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            pm = lambda k: f"{r[k + '_mean']:.3f} ± {0 if pd.isna(r[k + '_std']) else r[k + '_std']:.3f}"
            params = "-" if pd.isna(r.n_params_mean) else f"{int(r.n_params_mean):,}"
            name = f"**{m}**" if m == OUR_MODEL else m
            lines.append(f"| {name} | {r.family} | {pm('roc_auc')} | {pm('pr_auc')} | {pm('balanced_acc')} | "
                         f"{pm('mcc')} | {pm('f1')} | "
                         f"{r.precision_mean:.3f} | {r.recall_mean:.3f} | {r.fpr_mean:.3f} | "
                         f"{r.train_s_mean:.1f} | {r.latency_us_mean:.1f} | {params} |")
        lines.append("")
    if df_mask is not None:
        lines += ["## Mask-ratio ablation (SAFE-MAE, selected scorer fixed)", "",
                  "| Dataset | Mask ratio | ROC-AUC | Bal. acc. | F1 |", "|---|---|---|---|---|"]
        for (ds, r), g in df_mask.groupby(["dataset", "mask_ratio"]):
            lines.append(f"| {ds} | {r} | {g.roc_auc.mean():.3f} | {g.balanced_acc.mean():.3f} | {g.f1.mean():.3f} |")
        lines.append("")
    if df_woa is not None:
        lines += ["## WOA ablation (SAFE-MAE)", "", "| Dataset | Setting | ROC-AUC | Bal. acc. | F1 | Train (s) | Latency (µs/flow) |",
                  "|---|---|---|---|---|---|---|"]
        for (ds, st), g in df_woa.groupby(["dataset", "setting"], sort=False):
            lines.append(f"| {ds} | {st} | {g.roc_auc.mean():.3f} | {g.balanced_acc.mean():.3f} | {g.f1.mean():.3f} | "
                         f"{g.train_s.mean():.1f} | {g.latency_us.mean():.1f} |")
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="M3 model comparison & ablation study")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--epochs", type=int, default=safe.EPOCHS)
    ap.add_argument("--quick", action="store_true", help="1 seed, 10 epochs, short ablation")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--keep-leaky", action="store_true", help="do NOT drop label-leaking columns")
    ap.add_argument("--use-best", action="store_true",
                    help="use results/tuning/<dataset>/best_config.json (from tune_safe.py) for SAFE")
    ap.add_argument("--with-supervised", action="store_true", help="also run Random Forest reference")
    ap.add_argument("--merge", action="store_true",
                    help="keep previous results (results/comparison/*.csv) for datasets not run now")
    args = ap.parse_args()

    ratios = MASK_ABLATION_RATIOS
    if args.quick:
        args.seeds, args.epochs, ratios = args.seeds[:1], 10, [0.0, 0.2, 0.5]

    os.makedirs(OUT_DIR, exist_ok=True)
    order = [name for name, _, _ in build_models(args.epochs, with_supervised=args.with_supervised)]

    rows, mask_rows, woa_rows = [], [], []
    curves, dists, info = {}, {}, {}

    for ds in args.datasets:
        print("\n" + "=" * 70 + f"\n  DATASET: {ds}\n" + "=" * 70)
        data = prepare_dataset(ds, use_woa=True, drop_leaky=not args.keep_leaky)
        curves[ds] = {}
        if args.use_best:
            safe_cfg = load_best_config(ds)
            if args.quick:
                safe_cfg["epochs"] = min(safe_cfg["epochs"], args.epochs)
        else:
            safe_cfg = {**DEFAULT_SAFE_CONFIG, "epochs": args.epochs}

        for name, family, make in build_models(args.epochs, safe_cfg, args.with_supervised):
            if "supervised" in name and data["Xatk"] is None:
                continue
            for seed in args.seeds:
                det = make()
                m, s = evaluate(det, data, seed)
                rows.append({"dataset": ds, "model": name, "family": family, "seed": seed, **m})
                if seed == args.seeds[0]:
                    curves[ds][name] = (data["yte"], s)
                    if name == OUR_MODEL:
                        dists[ds] = (data["yte"], s, m["threshold"])
                        # fix the scorer chosen on validation for the ablations
                        kind, k = det.chosen
                        safe_cfg = {**safe_cfg, "scorer": kind, "lof_k": k}
                        info[ds] = {"attack_share": float(np.mean(data["yte"])), "scorer": m["scorer"]}
            last = [r for r in rows if r["dataset"] == ds and r["model"] == name]
            tag = f" [{last[0]['scorer']}]" if last[0].get("scorer") else ""
            print(f"  {name:<34} ROC-AUC {np.mean([r['roc_auc'] for r in last]):.4f} | "
                  f"BalAcc {np.mean([r['balanced_acc'] for r in last]):.4f} | "
                  f"MCC {np.mean([r['mcc'] for r in last]):.3f} | "
                  f"F1 {np.mean([r['f1'] for r in last]):.4f} | "
                  f"train {np.mean([r['train_s'] for r in last]):6.1f}s | "
                  f"{np.mean([r['latency_us'] for r in last]):7.1f} µs/flow{tag}")

        if not args.skip_ablation:
            print("\n  -- Mask-ratio ablation --")
            for r in ratios:
                for seed in args.seeds:
                    m, _ = evaluate(make_safe(safe_cfg, mask_ratio=r), data, seed)
                    mask_rows.append({"dataset": ds, "mask_ratio": r, "seed": seed, **m})
                print(f"  mask={r:.1f}  ROC-AUC {np.mean([x['roc_auc'] for x in mask_rows if x['dataset'] == ds and x['mask_ratio'] == r]):.4f}")

            print("\n  -- WOA vs all-features ablation --")
            n_woa = len(data["features"])
            for r in [x for x in rows if x["dataset"] == ds and x["model"] == OUR_MODEL]:
                woa_rows.append({"dataset": ds, "setting": f"WOA features ({n_woa})", **r})
            del data
            gc.collect()
            data_all = prepare_dataset(ds, use_woa=False, drop_leaky=not args.keep_leaky)
            n_all = len(data_all["features"])
            for seed in args.seeds:
                m, _ = evaluate(make_safe(safe_cfg), data_all, seed)
                woa_rows.append({"dataset": ds, "setting": f"All features ({n_all})", "seed": seed, **m})
            del data_all
        else:
            del data
        gc.collect()

    # ---------------- merge with previous run ----------------
    df = pd.DataFrame(rows)
    plot_datasets = list(args.datasets)
    if args.merge:
        def merged(new, name):
            path = os.path.join(OUT_DIR, name)
            if not os.path.exists(path):
                return new
            old = pd.read_csv(path)
            old = old[~old["dataset"].isin(args.datasets)]
            if len(old):
                print(f"[compare] --merge: kept {len(old)} previous rows from {name} "
                      f"for {sorted(old['dataset'].unique())}")
            return pd.concat([old, new], ignore_index=True) if new is not None else (old if len(old) else None)
        df = merged(df, "raw_results.csv")
        if "scorer" in df.columns:
            df["scorer"] = df["scorer"].fillna("")
        mask_rows = merged(pd.DataFrame(mask_rows) if mask_rows else None, "mask_ablation.csv")
        woa_rows = merged(pd.DataFrame(woa_rows) if woa_rows else None, "woa_ablation.csv")
        mask_rows = [] if mask_rows is None else mask_rows.to_dict("records")
        woa_rows = [] if woa_rows is None else woa_rows.to_dict("records")
        plot_datasets = [d for d in DATASETS if d in set(df["dataset"])] + \
                        [d for d in df["dataset"].unique() if d not in DATASETS]
        for d in plot_datasets:
            if d not in info:
                sub = df[(df.dataset == d) & (df.model == OUR_MODEL)]
                if len(sub) and "scorer" in sub:
                    info[d] = {"attack_share": float("nan"), "scorer": sub["scorer"].iloc[0]}

    # ---------------- save tables ----------------
    df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)
    num = ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr", "accuracy",
           "specificity", "balanced_acc", "mcc", "train_s", "latency_us", "n_params"]
    for c in num:
        if c not in df.columns:
            df[c] = np.nan
    agg = df.groupby(["dataset", "model", "family"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "summary_results.csv"), index=False)

    df_mask = pd.DataFrame(mask_rows) if mask_rows else None
    df_woa = pd.DataFrame(woa_rows) if woa_rows else None
    if df_mask is not None:
        df_mask.to_csv(os.path.join(OUT_DIR, "mask_ablation.csv"), index=False)
    if df_woa is not None:
        df_woa.to_csv(os.path.join(OUT_DIR, "woa_ablation.csv"), index=False)

    # ---------------- figures ----------------
    plot_metric_bars(agg, plot_datasets, order)
    for ds in args.datasets:
        plot_curves(curves[ds], ds, order)
    plot_efficiency(agg, plot_datasets)
    if df_mask is not None:
        plot_mask_ablation(df_mask)
    if df_woa is not None:
        plot_woa_ablation(df_woa)
    plot_score_distribution(dists)
    write_summary_md(agg, df_mask, df_woa, plot_datasets, order, info)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"seeds": args.seeds, "epochs": args.epochs, "mask_ratio": MASK_RATIO,
                   "dropped_leaky_columns": not args.keep_leaky, "use_best": args.use_best,
                   "max_normal_train": safe.MAX_NORMAL_TRAIN_SAMPLES,
                   "max_test": safe.MAX_TEST_SAMPLES}, f, indent=2)

    print("\n" + "=" * 70 + f"\n  Done. Tables and figures saved to: {OUT_DIR}\n" + "=" * 70)


if __name__ == "__main__":
    main()