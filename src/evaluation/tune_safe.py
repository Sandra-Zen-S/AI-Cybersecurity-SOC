"""
SAFE Hyperparameter Study (epochs, mask ratio, learning rate, batch size,
latent size, dropout, loss) on IoMT-TrafficData and MedSec-25
==========================================================================

Why this exists: to justify every SAFE setting with evidence ("why 30 epochs
and not 60?") instead of picking values by hand.

How selection works (no test-set leakage):
  * Every setting is SELECTED on the VALIDATION set only
      - normal validation split (20% of normal training flows)
      - attack validation sample (from the TRAINING csv, as in safe.py)
    metric = validation ROC-AUC (threshold-free).
  * Test metrics are recorded for reporting, never used to choose.
  * Parsimony rule: if a cheaper/simpler value is within TOLERANCE of the
    best validation ROC-AUC, the cheaper one wins (e.g. 30 epochs beats 80
    epochs if 80 gains < 0.002 AUC).

Studies
  1. Epoch study   : one training run per seed up to --max-epochs, evaluated
                     after EVERY epoch (train loss, val loss, val AUC, test AUC,
                     test F1, cumulative train time). Gives the full 10/20/30/
                     .../100 curve from a single run, plus an early-stopping
                     marker (val loss patience).
  2. Sensitivity   : one-at-a-time grid around the defaults for mask ratio,
                     learning rate, batch size, latent size, dropout, loss.
  3. Confirmation  : default config vs tuned config, several seeds, test set.

Outputs (results/tuning/)
  <dataset>/epoch_history.csv, <dataset>/sensitivity.csv,
  <dataset>/best_config.json, final_default_vs_tuned.csv, summary_tuning.md,
  fig_epochs_<dataset>.png, fig_sensitivity.png, fig_default_vs_tuned.png

Usage (from the project root, after placing this file in src/evaluation/):
  python -m src.evaluation.tune_safe --quick         # smoke test (~10-15 min)
  python -m src.evaluation.tune_safe                 # full study
  python -m src.evaluation.tune_safe --datasets MedSec-25
  python -m src.evaluation.tune_safe --skip-grid     # epoch study only
Then:
  python -m src.evaluation.compare_models --use-best
"""

import os
import gc
import json
import time
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from .compare_models import (
    prepare_dataset, pick_threshold, ConfigurableAE, DEFAULT_SAFE_CONFIG,
    PROJECT_ROOT, DATASETS,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "tuning")
DEVICE = torch.device("cpu")

TOLERANCE = 0.002          # val ROC-AUC difference treated as "no real gain"
ES_PATIENCE = 5            # early-stopping patience on validation loss (epochs)
EPOCH_MARKS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]

GRID = {
    "mask_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr":         [1e-4, 5e-4, 1e-3, 3e-3],
    "batch_size": [64, 128, 256, 512],
    "latent":     [8, 16, 32, 64],
    "dropout":    [0.0, 0.1, 0.2, 0.3],
    "loss":       ["huber", "mse"],
}
QUICK_GRID = {
    "mask_ratio": [0.0, 0.2, 0.4],
    "lr":         [1e-4, 5e-4, 3e-3],
    "batch_size": [128, 512],
    "latent":     [16, 32],
    "dropout":    [0.0, 0.1],
    "loss":       ["huber", "mse"],
}
# when two values tie within TOLERANCE, prefer the one listed earlier here
SIMPLER_FIRST = {
    "mask_ratio": None,                       # tie -> keep default
    "lr": None,
    "batch_size": [512, 256, 128, 64],        # bigger batch = faster
    "latent": [8, 16, 32, 64],                # smaller = simpler
    "dropout": None,
    "loss": None,
}


# ============================================================
# TRAINING WITH PER-EPOCH EVALUATION
# ============================================================

def recon_errors(net, X, batch=4096):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch])
            out.append(((net(b) - b) ** 2).mean(1).numpy())
    return np.concatenate(out)


def eval_point(net, data, loss_fn):
    s_val_n = recon_errors(net, data["Xval"])
    s_val_a = recon_errors(net, data["Xatk"]) if data["Xatk"] is not None else None
    s_te = recon_errors(net, data["Xte"])

    with torch.no_grad():
        xv = torch.from_numpy(data["Xval"])
        val_loss = float(loss_fn(net(xv), xv))

    val_auc = (roc_auc_score(np.r_[np.zeros(len(s_val_n)), np.ones(len(s_val_a))], np.r_[s_val_n, s_val_a])
               if s_val_a is not None else np.nan)
    thr = pick_threshold(s_val_n, s_val_a)
    y = data["yte"]
    pred = (s_te > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "val_loss": val_loss, "val_auc": val_auc,
        "test_auc": roc_auc_score(y, s_te), "test_pr_auc": average_precision_score(y, s_te),
        "test_f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "test_precision": prec, "test_recall": rec, "test_fpr": fp / max(fp + tn, 1),
    }


def train_safe(data, cfg, seed, max_epochs, eval_every_epoch=False):
    """Train SAFE with config `cfg`. Returns list of per-epoch (or final) metric dicts."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.from_numpy(data["Xtr"])
    net = ConfigurableAE(X.shape[1], latent=cfg["latent"], dropout=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(X), batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(seed))

    history, train_time = [], 0.0
    for ep in range(1, max_epochs + 1):
        net.train()
        t0, losses = time.perf_counter(), []
        for (b,) in loader:
            inp = b
            if cfg["mask_ratio"] > 0:
                inp = b.clone()
                inp[torch.rand_like(b) < cfg["mask_ratio"]] = 0.0
            loss = loss_fn(net(inp), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_time += time.perf_counter() - t0

        if eval_every_epoch or ep == max_epochs:
            point = eval_point(net, data, loss_fn)
            history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                            "train_time_s": train_time, **point})
    return history


# ============================================================
# SELECTION HELPERS
# ============================================================

def choose_epochs(hist_df):
    """Smallest epoch whose (5-epoch smoothed) mean val AUC is within TOLERANCE
    of the best. Smoothing stops a single lucky epoch from being selected."""
    g = hist_df.groupby("epoch")["val_auc"].mean().rolling(5, center=True, min_periods=1).mean()
    best = g.max()
    return int(g[g >= best - TOLERANCE].index.min()), float(best)


def early_stop_epoch(hist_df):
    """Epoch where mean val loss stopped improving (patience ES_PATIENCE)."""
    g = hist_df.groupby("epoch")["val_loss"].mean()
    best, best_ep, wait = np.inf, int(g.index.min()), 0
    for ep, v in g.items():
        if v < best - 1e-6:
            best, best_ep, wait = v, int(ep), 0
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                break
    return best_ep


def choose_value(param, sens_df, default_value):
    sub = sens_df[sens_df.param == param]
    g = sub.groupby("value_str")["val_auc"].mean()
    best_val = g.max()
    default_key = str(default_value)
    # Keep the default unless something beats it by more than TOLERANCE
    if default_key in g.index and g[default_key] >= best_val - TOLERANCE:
        candidates = [v for v in g.index if g[v] >= best_val - TOLERANCE]
        order = SIMPLER_FIRST.get(param)
        if order:
            for v in order:
                if str(v) in candidates:
                    return v
        return default_value
    winner = g.idxmax()
    return next(v for v in sub.value if str(v) == winner)


# ============================================================
# PLOTS
# ============================================================

def plot_epochs(hist_df, ds, chosen, es_ep):
    g = hist_df.groupby("epoch").agg(["mean", "std"])
    ep = g.index.values

    def band(ax, col, label, **kw):
        m, s = g[(col, "mean")].values, g[(col, "std")].fillna(0).values
        ax.plot(ep, m, label=label, **kw)
        ax.fill_between(ep, m - s, m + s, alpha=0.18, color=kw.get("color"))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    band(axes[0], "train_loss", "Train loss", color="#1f77b4")
    band(axes[0], "val_loss", "Validation loss (normal)", color="#ff7f0e")
    axes[0].set(title="Reconstruction loss", xlabel="Epoch", ylabel="Loss")
    axes[0].set_yscale("log")

    band(axes[1], "val_auc", "Validation ROC-AUC (selection)", color="#2ca02c")
    band(axes[1], "test_auc", "Test ROC-AUC (report only)", color="#d62728", ls="--")
    band(axes[1], "test_f1", "Test F1 (report only)", color="#9467bd", ls=":")
    axes[1].set(title="Detection quality vs. epochs", xlabel="Epoch", ylabel="Score")

    band(axes[2], "train_time_s", "Cumulative training time", color="#7f7f7f")
    axes[2].set(title="Training cost", xlabel="Epoch", ylabel="Seconds")

    for ax in axes:
        ax.axvline(30, color="k", ls=":", lw=1, label="safe.py default (30)")
        ax.axvline(chosen, color="#2ca02c", ls="--", lw=1.4, label=f"selected ({chosen})")
        ax.axvline(es_ep, color="#ff7f0e", ls="-.", lw=1, label=f"early-stop on val loss ({es_ep})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{ds}: SAFE epoch study (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig_epochs_{ds}.png"), dpi=200)
    plt.close(fig)


def plot_sensitivity(sens_all, grid, defaults_by_ds, chosen_by_ds):
    params = list(grid.keys())
    cols = 3
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.4 * rows), squeeze=False)
    for ax, p in zip(axes.flat, params):
        vals = grid[p]
        x = np.arange(len(vals))
        for ds, sens in sens_all.items():
            sub = sens[sens.param == p]
            g = sub.groupby("value_str")[["val_auc", "test_auc"]].agg(["mean", "std"])
            m = [g.loc[str(v), ("val_auc", "mean")] if str(v) in g.index else np.nan for v in vals]
            s = [g.loc[str(v), ("val_auc", "std")] if str(v) in g.index else np.nan for v in vals]
            t = [g.loc[str(v), ("test_auc", "mean")] if str(v) in g.index else np.nan for v in vals]
            line = ax.errorbar(x, m, yerr=np.nan_to_num(s), marker="o", capsize=3, label=f"{ds} (val)")
            ax.plot(x, t, ls="--", alpha=0.6, color=line[0].get_color(), label=f"{ds} (test)")
            ci = [str(v) for v in vals].index(str(chosen_by_ds[ds][p]))
            ax.scatter([x[ci]], [m[ci]], s=160, facecolors="none", edgecolors=line[0].get_color(), lw=2)
        di = [str(v) for v in vals].index(str(DEFAULT_SAFE_CONFIG[p]))
        ax.axvspan(di - 0.3, di + 0.3, color="grey", alpha=0.12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set(title=f"{p}  (grey = safe.py default, circle = selected)", ylabel="ROC-AUC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    fig.suptitle("SAFE one-at-a-time sensitivity (selected on validation ROC-AUC)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_sensitivity.png"), dpi=200)
    plt.close(fig)


def plot_final(final_df):
    metrics = [("test_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC"), ("test_f1", "F1"), ("train_time_s", "Train time (s)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (m, title) in zip(axes, metrics):
        g = final_df.groupby(["dataset", "config"])[m].agg(["mean", "std"]).reset_index()
        dss = g.dataset.unique()
        x, w = np.arange(len(dss)), 0.38
        for i, c in enumerate(["default", "tuned"]):
            sub = g[g.config == c].set_index("dataset").loc[dss]
            ax.bar(x + (i - 0.5) * w, sub["mean"], w, yerr=sub["std"].fillna(0), capsize=3, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(dss)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SAFE: safe.py defaults vs tuned configuration (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_default_vs_tuned.png"), dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="SAFE hyperparameter study")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="1 seed, 40 epochs, small grid")
    ap.add_argument("--skip-epochs", action="store_true", help="skip epoch study (uses 30 epochs)")
    ap.add_argument("--skip-grid", action="store_true", help="skip sensitivity grid")
    ap.add_argument("--keep-leaky", action="store_true", help="do NOT drop label-leaking columns")
    args = ap.parse_args()

    grid = GRID
    if args.quick:
        args.seeds, args.max_epochs, grid = args.seeds[:1], 40, QUICK_GRID
    os.makedirs(OUT_DIR, exist_ok=True)

    sens_all, chosen_all, final_rows, md = {}, {}, [], ["# SAFE Hyperparameter Study", ""]
    md += [f"Selection metric: validation ROC-AUC (normal val split + attack val sample from train). "
           f"Tolerance {TOLERANCE}: a cheaper/simpler value within this margin of the best is preferred. "
           f"Seeds: {args.seeds}.", ""]

    for ds in args.datasets:
        print("\n" + "=" * 70 + f"\n  SAFE TUNING: {ds}\n" + "=" * 70)
        ds_dir = os.path.join(OUT_DIR, ds)
        os.makedirs(ds_dir, exist_ok=True)
        data = prepare_dataset(ds, use_woa=True, drop_leaky=not args.keep_leaky)
        if data["Xatk"] is None:
            print(f"[tune] No attack validation samples for {ds}; cannot compute validation ROC-AUC. Skipping.")
            continue
        md += [f"## {ds}", ""]
        tuned = dict(DEFAULT_SAFE_CONFIG)

        # ---------------- 1. epoch study ----------------
        if not args.skip_epochs:
            print(f"\n  -- Epoch study (1..{args.max_epochs}, evaluated every epoch) --")
            rows = []
            for seed in args.seeds:
                cfg = {**DEFAULT_SAFE_CONFIG}
                for h in train_safe(data, cfg, seed, args.max_epochs, eval_every_epoch=True):
                    rows.append({"seed": seed, **h})
                print(f"    seed {seed} done")
            hist = pd.DataFrame(rows)
            hist.to_csv(os.path.join(ds_dir, "epoch_history.csv"), index=False)
            chosen_ep, best_auc = choose_epochs(hist)
            es_ep = early_stop_epoch(hist)
            tuned["epochs"] = chosen_ep
            plot_epochs(hist, ds, chosen_ep, es_ep)

            g = hist.groupby("epoch").mean(numeric_only=True)
            marks = [e for e in EPOCH_MARKS if e in g.index]
            md += ["### Epoch study", "",
                   "| Epochs | Train loss | Val loss | Val ROC-AUC | Test ROC-AUC | Test F1 | Train time (s) |",
                   "|---|---|---|---|---|---|---|"]
            for e in marks:
                r = g.loc[e]
                md.append(f"| {e} | {r.train_loss:.5f} | {r.val_loss:.5f} | {r.val_auc:.4f} | "
                          f"{r.test_auc:.4f} | {r.test_f1:.4f} | {r.train_time_s:.1f} |")
            md += ["", f"Best mean val ROC-AUC {best_auc:.4f}; **selected epochs = {chosen_ep}** "
                   f"(smallest within {TOLERANCE}). Early stopping on val loss would stop at epoch {es_ep}.", ""]
            print(f"    selected epochs = {chosen_ep} (best val AUC {best_auc:.4f}), early-stop epoch {es_ep}")

        # ---------------- 2. sensitivity grid ----------------
        if not args.skip_grid:
            print(f"\n  -- Sensitivity grid ({tuned['epochs']} epochs) --")
            cache, rows = {}, []
            base = {**DEFAULT_SAFE_CONFIG, "epochs": tuned["epochs"]}
            for p, values in grid.items():
                for v in values:
                    cfg = {**base, p: v}
                    key = json.dumps(cfg, sort_keys=True)
                    for seed in args.seeds:
                        if (key, seed) not in cache:
                            cache[(key, seed)] = train_safe(data, cfg, seed, cfg["epochs"])[-1]
                        rows.append({"param": p, "value": v, "value_str": str(v), "seed": seed,
                                     **cache[(key, seed)]})
                    last = [r for r in rows if r["param"] == p and r["value_str"] == str(v)]
                    print(f"    {p:<11}= {str(v):<7} val AUC {np.mean([r['val_auc'] for r in last]):.4f} | "
                          f"test AUC {np.mean([r['test_auc'] for r in last]):.4f}")
            sens = pd.DataFrame(rows)
            sens.to_csv(os.path.join(ds_dir, "sensitivity.csv"), index=False)
            sens_all[ds] = sens

            md += ["### Sensitivity (one-at-a-time)", "",
                   "| Parameter | Value | Val ROC-AUC | Test ROC-AUC | Test F1 | Train time (s) |",
                   "|---|---|---|---|---|---|"]
            for p in grid:
                tuned[p] = choose_value(p, sens, DEFAULT_SAFE_CONFIG[p])
                g = sens[sens.param == p].groupby("value_str").mean(numeric_only=True)
                for v in grid[p]:
                    if str(v) not in g.index:
                        continue
                    r = g.loc[str(v)]
                    tag = " **(selected)**" if str(v) == str(tuned[p]) else ""
                    md.append(f"| {p} | {v}{tag} | {r.val_auc:.4f} | {r.test_auc:.4f} | "
                              f"{r.test_f1:.4f} | {r.train_time_s:.1f} |")
            md.append("")
        chosen_all[ds] = tuned

        # ---------------- 3. default vs tuned ----------------
        print(f"\n  -- Confirmation: default vs tuned {tuned} --")
        default_cfg = dict(DEFAULT_SAFE_CONFIG)
        for name, cfg in [("default", default_cfg), ("tuned", tuned)]:
            for seed in args.seeds:
                r = train_safe(data, cfg, seed, cfg["epochs"])[-1]
                final_rows.append({"dataset": ds, "config": name, "seed": seed, **r})
        fd = pd.DataFrame([r for r in final_rows if r["dataset"] == ds]).drop(columns=["dataset"])
        g = fd.groupby("config").agg(["mean", "std"])
        # The one-at-a-time winners are not guaranteed to work best together:
        # keep the combined config only if it also wins on VALIDATION ROC-AUC.
        val_default, val_tuned = g.loc["default", ("val_auc", "mean")], g.loc["tuned", ("val_auc", "mean")]
        final_cfg = tuned if val_tuned > val_default + TOLERANCE else default_cfg
        final_name = "tuned" if final_cfg is tuned else "default"
        md += ["### Default vs tuned (test set)", "",
               f"Default: `{default_cfg}`  ", f"Tuned: `{tuned}`", "",
               f"Validation ROC-AUC: default {val_default:.4f}, tuned {val_tuned:.4f} "
               f"-> **final choice: {final_name}**", "",
               "| Config | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Precision | Recall | FPR | Train time (s) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for c in ["default", "tuned"]:
            pm = lambda k: f"{g.loc[c, (k, 'mean')]:.4f} ± {np.nan_to_num(g.loc[c, (k, 'std')]):.4f}"
            md.append(f"| {c} | {pm('val_auc')} | {pm('test_auc')} | {pm('test_pr_auc')} | {pm('test_f1')} | "
                      f"{g.loc[c, ('test_precision', 'mean')]:.4f} | {g.loc[c, ('test_recall', 'mean')]:.4f} | "
                      f"{g.loc[c, ('test_fpr', 'mean')]:.4f} | {g.loc[c, ('train_time_s', 'mean')]:.1f} |")
        md.append("")

        with open(os.path.join(ds_dir, "best_config.json"), "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "selection": "validation ROC-AUC", "tolerance": TOLERANCE,
                       "seeds": args.seeds, "default_config": default_cfg,
                       "one_at_a_time_config": tuned, "final_choice": final_name,
                       "tuned_config": final_cfg}, f, indent=2)
        chosen_all[ds] = tuned
        print(f"    final choice: {final_name} -> {final_cfg}")
        print(f"    saved {os.path.join(ds_dir, 'best_config.json')}")
        del data
        gc.collect()

    if sens_all:
        plot_sensitivity(sens_all, grid, DEFAULT_SAFE_CONFIG, chosen_all)
    if final_rows:
        fdf = pd.DataFrame(final_rows)
        fdf.to_csv(os.path.join(OUT_DIR, "final_default_vs_tuned.csv"), index=False)
        plot_final(fdf)
    with open(os.path.join(OUT_DIR, "summary_tuning.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "=" * 70 + f"\n  Done. Results in: {OUT_DIR}\n" + "=" * 70)


if __name__ == "__main__":
    main()