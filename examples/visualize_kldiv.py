#!/usr/bin/env python3
"""
Visualize Momentum+KL LeetCode Experiment Results

This script is tailored for the momentum_kldiv_dataset experiment:
- Metrics supported out-of-the-box:
  - momentum_loss_mean           (training objective)
  - kl_C0_Ct_mean                (monitoring, KL from init to current)
  - kl_Ctm1_Ct_mean              (monitoring, KL from prev-step to current; spikes indicate learning jumps)
  - lm_entropy_mean              (optional PoE, average next-token entropy of LM)
  - temperature                  (training temperature per step)

Features:
- Load runs under results/<timestamp>/init=...__temp=...__sched=...__seed=.../
- Filter on any cfg_* config field (init_strategy, temperature, schedule, seed, poe_gamma, kl_clip, etc.)
- Grouping, hue, faceting (per-metric)
- Smoothing (EWMA, moving average)
- Overlay temperature on a secondary axis
- Annotate spikes on kl_Ctm1_Ct_mean (threshold-based)
- Export aggregated stats to CSV

Examples:

  python examples/visualize_momentum_kldiv.py \
    --root ./results/momentum_kldiv_ds/20250919_120000 \
    --metrics momentum_loss_mean kl_C0_Ct_mean kl_Ctm1_Ct_mean \
    --group_by cfg_temperature \
    --hue cfg_init_strategy \
    --facet cfg_schedule \
    --smooth ewma --alpha 0.2 \
    --annotate_spikes --spike_metric kl_Ctm1_Ct_mean --spike_threshold 1.0 \
    --overlay_temperature \
    --save_dir ./figs/momentum_kldiv

  python examples/visualize_momentum_kldiv.py \
    --root ./results/momentum_kldiv_ds \
    --recursive \
    --filters cfg_init_strategy=fluency cfg_schedule=cosine \
    --metrics kl_C0_Ct_mean \
    --style seaborn-darkgrid
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------- Discovery & Loading -------------------------- #

def find_runs(root: Path, recursive: bool = False) -> List[Path]:
    """
    Discover run directories containing config.json and trace.csv
    """
    if recursive:
        cfgs = list(root.rglob("config.json"))
        runs = [p.parent for p in cfgs if (p.parent / "trace.csv").exists()]
    else:
        runs = [p for p in root.glob("*") if p.is_dir() and (p / "config.json").exists() and (p / "trace.csv").exists()]
    return sorted(runs)


def load_run(run_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    with open(run_dir / "config.json", "r") as f:
        cfg = json.load(f)
    df = pd.read_csv(run_dir / "trace.csv")
    for k, v in cfg.items():
        df[f"cfg_{k}"] = v
    df["run_dir"] = str(run_dir)
    return cfg, df


def load_runs(root: Path, recursive: bool = False) -> pd.DataFrame:
    runs = find_runs(root, recursive=recursive)
    if not runs:
        return pd.DataFrame()
    frames = []
    for r in runs:
        try:
            _, df = load_run(r)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {r}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -------------------------- Filtering & Smoothing ------------------------ #

def parse_filters(filter_list: List[str]) -> Dict[str, List[str]]:
    """
    Parse CLI filters of the form: key=value or key=value1,value2
    Returns a dict of {key: [values...]}
    """
    result: Dict[str, List[str]] = {}
    for spec in filter_list:
        if "=" not in spec:
            print(f"[WARN] Ignoring malformed filter '{spec}', expected key=value")
            continue
        k, v = spec.split("=", 1)
        vals = [s.strip() for s in v.split(",") if s.strip() != ""]
        if vals:
            result[k] = vals
    return result


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    if df.empty or not filters:
        return df
    mask = np.ones(len(df), dtype=bool)
    for k, vals in filters.items():
        if k not in df.columns:
            print(f"[WARN] Filter key '{k}' not in columns; skipping this filter.")
            continue
        col = df[k].astype(str)
        mask &= col.isin([str(v) for v in vals])
    return df[mask].copy()


def ewma(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def smooth_series(y: np.ndarray, mode: Optional[str], alpha: float, window: int) -> np.ndarray:
    if mode is None or mode == "none":
        return y
    if mode == "ewma":
        return ewma(y, alpha)
    if mode == "moving_avg":
        return moving_average(y, window)
    return y


# ------------------------------ Plotting --------------------------------- #

def plot_metric_grid(
    df: pd.DataFrame,
    metric: str,
    x: str = "step",
    group_by: Optional[str] = "cfg_temperature",
    hue: Optional[str] = "cfg_init_strategy",
    facet: Optional[str] = "cfg_schedule",
    smooth: Optional[str] = "ewma",
    alpha: float = 0.2,
    mov_window: int = 51,
    style: str = "seaborn-darkgrid",
    sharey: bool = False,
    annotate_spikes: bool = False,
    spike_metric: str = "kl_Ctm1_Ct_mean",
    spike_threshold: float = 1.0,
    overlay_temperature: bool = False,
    figsize: Tuple[int, int] = (8, 5),
    legend: bool = True,
):
    if df.empty:
        print("[INFO] No data to plot.")
        return

    plt.style.use(style)
    facets = sorted(df[facet].unique()) if facet and facet in df.columns else [None]
    ncols = min(3, len(facets))
    nrows = int(np.ceil(len(facets) / max(1, ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*max(1,ncols), figsize[1]*max(1,nrows)), squeeze=False, sharey=sharey)
    axes = axes.flatten()

    for i, fval in enumerate(facets):
        ax = axes[i]
        df_f = df if fval is None else df[df[facet] == fval]

        groups = sorted(df_f[group_by].unique()) if group_by and group_by in df_f.columns else [None]
        hues = sorted(df_f[hue].unique()) if hue and hue in df_f.columns else [None]

        for g in groups:
            df_g = df_f if g is None else df_f[df_f[group_by] == g]
            for h in hues:
                df_h = df_g if h is None else df_g[df_g[hue] == h]
                if df_h.empty:
                    continue

                # Aggregate across seeds/runs: mean/std per x
                agg = df_h.groupby(x)[metric].agg(["mean", "std"]).reset_index()
                X = agg[x].values
                Y = agg["mean"].values
                Yp = smooth_series(Y, smooth, alpha, mov_window)

                label = []
                if g is not None: label.append(f"{group_by}={g}")
                if h is not None: label.append(f"{hue}={h}")
                label = ", ".join(label) if label else "run"

                ax.plot(X, Yp, label=label)
                # std shading (optional)
                # ax.fill_between(X, Yp - agg["std"].values, Yp + agg["std"].values, alpha=0.15)

                # Spike annotations (based on spike_metric)
                if annotate_spikes and spike_metric in df_h.columns and spike_threshold is not None:
                    agg_spk = df_h.groupby(x)[spike_metric].mean().reset_index()
                    spikes = agg_spk[agg_spk[spike_metric] >= spike_threshold][x].values
                    for s in spikes:
                        ax.axvline(s, color="red", alpha=0.15, linestyle="--")

        if overlay_temperature and "temperature" in df_f.columns:
            ax2 = ax.twinx()
            temp_agg = df_f.groupby(x)["temperature"].mean().reset_index()
            ax2.plot(temp_agg[x].values, temp_agg["temperature"].values, color="gray", alpha=0.4, linestyle=":", label="temperature")
            ax2.set_ylabel("temperature", color="gray")
            ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_xlabel(x)
        ax.set_ylabel(metric)
        ttl = f"{facet}={fval}" if fval is not None else metric
        ax.set_title(ttl)
        if legend:
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    return fig, axes


# --------------------------- Aggregation & Export ------------------------ #

def export_aggregate_stats(df: pd.DataFrame, out_csv: Optional[str], metrics: List[str], xcol: str = "step"):
    if not out_csv:
        return
    rows = []
    # For each run, compute final values and AUC-like sums per metric
    for run_dir, df_run in df.groupby("run_dir"):
        row = {"run_dir": run_dir}
        # include config fields (once per run)
        cfg_cols = [c for c in df_run.columns if c.startswith("cfg_")]
        for c in cfg_cols:
            row[c] = df_run[c].iloc[0]
        # final values and sums
        max_step = df_run[xcol].max()
        for m in metrics:
            if m in df_run.columns:
                row[f"{m}_final"] = df_run.loc[df_run[xcol].idxmax(), m]
                # trapezoidal sum (rough AUC-like over steps)
                dfr = df_run.sort_values(by=xcol)
                row[f"{m}_sum"] = float(np.trapz(dfr[m].values, dfr[xcol].values))
        rows.append(row)
    agg = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"[INFO] Exported aggregate stats: {out_csv}")


# --------------------------------- CLI ---------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to a timestamped results directory OR a parent directory containing runs")
    ap.add_argument("--recursive", action="store_true", help="Search recursively for runs")
    # Filters: ex: --filters cfg_init_strategy=fluency cfg_schedule=cosine cfg_temperature=100
    ap.add_argument("--filters", type=str, nargs="*", default=[], help="key=value pairs; multiple values comma-separated")
    # Selection
    ap.add_argument("--metrics", type=str, nargs="+",
                    default=["momentum_loss_mean", "kl_C0_Ct_mean", "kl_Ctm1_Ct_mean"],
                    help="Metrics to plot")
    ap.add_argument("--x", type=str, default="step")
    ap.add_argument("--group_by", type=str, default="cfg_temperature")
    ap.add_argument("--hue", type=str, default="cfg_init_strategy")
    ap.add_argument("--facet", type=str, default="cfg_schedule")
    # Smoothing
    ap.add_argument("--smooth", type=str, default="ewma", choices=["none", "ewma", "moving_avg"])
    ap.add_argument("--alpha", type=float, default=0.2, help="EWMA alpha")
    ap.add_argument("--mov_window", type=int, default=51, help="Moving average window")
    # Plot style
    ap.add_argument("--style", type=str, default="seaborn-darkgrid")
    ap.add_argument("--sharey", action="store_true")
    ap.add_argument("--legend", action="store_true")
    ap.add_argument("--fig_w", type=int, default=8)
    ap.add_argument("--fig_h", type=int, default=5)
    # Extras
    ap.add_argument("--annotate_spikes", action="store_true", help="Mark vertical lines where spike_metric exceeds threshold")
    ap.add_argument("--spike_metric", type=str, default="kl_Ctm1_Ct_mean")
    ap.add_argument("--spike_threshold", type=float, default=1.0)
    ap.add_argument("--overlay_temperature", action="store_true")
    # Saving
    ap.add_argument("--save_dir", type=str, default=None, help="Save plots here (one PNG per metric). If None, just show.")
    ap.add_argument("--export_csv", type=str, default=None, help="Optional: export aggregated stats to CSV")
    args = ap.parse_args()

    root = Path(args.root)
    df = load_runs(root, recursive=args.recursive)
    if df.empty:
        print("[INFO] No runs found.")
        return

    filters = parse_filters(args.filters)
    df = apply_filters(df, filters)
    if df.empty:
        print("[INFO] No data left after applying filters.")
        return

    # Export aggregate stats (optional)
    export_aggregate_stats(df, args.export_csv, metrics=args.metrics, xcol=args.x)

    # Plot each requested metric
    out_dir = Path(args.save_dir) if args.save_dir else None
    for metric in args.metrics:
        fig, axes = plot_metric_grid(
            df=df,
            metric=metric,
            x=args.x,
            group_by=args.group_by,
            hue=args.hue,
            facet=args.facet,
            smooth=args.smooth,
            alpha=args.alpha,
            mov_window=args.mov_window,
            style=args.style,
            sharey=args.sharey,
            annotate_spikes=args.annotate_spikes,
            spike_metric=args.spike_metric,
            spike_threshold=args.spike_threshold,
            overlay_temperature=args.overlay_temperature,
            figsize=(args.fig_w, args.fig_h),
            legend=args.legend,
        )
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{metric}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved: {out_path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
