#!/usr/bin/env python3
"""
scripts/generate_figures.py

Generates publication-quality figures FROM REAL DATA produced by
scripts/run_evaluation.py. This script contains no hardcoded numbers —
every figure is computed from the CSV/JSON files you point it at.

For each system you evaluated, run scripts/run_evaluation.py once
(writing results/<system>_per_example.csv and
results/<system>_summary.json), then run this script pointed at the
results/ directory.

Output formats (both written for every figure):
  - Vector PDF and SVG — what you actually want for a Springer LNCS
    submission. Vector graphics scale losslessly to any print
    resolution; conferences and typesetters strongly prefer them over
    raster images.
  - High-resolution PNG at true ~4K pixel density (long edge ~3840px)
    — useful for slides, READMEs, or venues that require raster.

Figures produced (5 total, regardless of how many systems you load —
multi-system figures are combined into ONE file each, not one per
system):
  1. bar_comparison             — Over-generation Rate / False Positive
                                   Rate / F1 across all evaluated
                                   systems (Table 1 visualized)
  2. sas_distribution            — SAS score distribution by trap type
                                   (box/strip plot)
  3. vdv_distribution            — VDV deception score distribution by
                                   trap type
  4. confusion_matrix_all_systems — one multi-panel figure, one panel
                                   per system, predicted-flagged vs.
                                   ground-truth
  5. threshold_curve_all_systems  — one figure, all systems overlaid as
                                   separate F1-vs-threshold lines

Usage:
    python scripts/generate_figures.py --results-dir results --out-dir figures

If results-dir contains only demo data (is_demo_data=true in the
summary JSON), every figure is stamped with a visible "DEMO DATA —
NOT FOR PUBLICATION" watermark so it can't accidentally end up in a
submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Style: clean, print-friendly, colorblind-safe palette
# --------------------------------------------------------------------------- #

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",  # keep text as text, not paths, in SVG output
})

_PALETTE = ["#2E5EAA", "#D9541E", "#3C9E3C", "#8A4FBF", "#C2185B", "#5C6B73"]


def _save_all_formats(fig: plt.Figure, out_dir: Path, name: str, is_demo: bool):
    if is_demo:
        fig.text(
            0.5, 0.5, "DEMO DATA — NOT FOR PUBLICATION",
            fontsize=28, color="red", alpha=0.35, ha="center", va="center",
            rotation=30, transform=fig.transFigure, zorder=100,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Vector formats first (what the Springer LNCS submission should
    # actually use): lossless at any print size.
    for ext in ("pdf", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    # PNG at true 4K pixel density (3840x2160 or larger, depending on the
    # figure's native aspect ratio) for slides/README/anywhere raster is
    # required. Computed per-figure so aspect ratio isn't distorted.
    png_path = out_dir / f"{name}.png"
    fig_w_in, fig_h_in = fig.get_size_inches()
    target_long_edge_px = 3840
    long_edge_in = max(fig_w_in, fig_h_in)
    dpi_for_4k = max(300, int(target_long_edge_px / long_edge_in))
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi_for_4k)
    print(f"  wrote {png_path} (~{int(fig_w_in*dpi_for_4k)}x{int(fig_h_in*dpi_for_4k)}px)")
    plt.close(fig)


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load every (per_example.csv, summary.json) pair found in
    results_dir, keyed by system name."""
    systems = {}
    for summary_path in sorted(results_dir.glob("*_summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        csv_path = results_dir / f"{summary_path.stem.replace('_summary', '')}_per_example.csv"
        if not csv_path.exists():
            print(f"  WARNING: no matching CSV for {summary_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        systems[summary["system_name"]] = {"summary": summary, "df": df}
    return systems


# --------------------------------------------------------------------------- #
# Figure 1 — bar comparison across systems
# --------------------------------------------------------------------------- #


def fig_bar_comparison(systems: Dict[str, Dict], out_dir: Path):
    names = list(systems.keys())
    if not names:
        print("  no systems loaded, skipping bar_comparison")
        return
    metrics = ["over_generation_rate", "false_positive_rate", "hybrid_f1"]
    metric_labels = ["Over-generation\nRate ↓", "False Positive\nRate ↓", "F1 ↑"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(names))

    is_demo = any(systems[n]["summary"].get("is_demo_data") for n in names)

    for i, name in enumerate(names):
        values = [systems[name]["summary"].get(m, float("nan")) for m in metrics]
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=_PALETTE[i % len(_PALETTE)])
        for b, v in zip(bars, values):
            if v == v:  # not NaN
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("System comparison on the adversarial evaluation set")
    ax.legend(loc="upper right", frameon=False)
    _save_all_formats(fig, out_dir, "bar_comparison", is_demo)


# --------------------------------------------------------------------------- #
# Figure 2 / 3 — SAS / VDV distributions by trap type
# --------------------------------------------------------------------------- #


def _distribution_figure(systems: Dict[str, Dict], out_dir: Path, column: str, title: str, fname: str):
    # Use the first system's per-example data (symbolic scores don't
    # depend on which LLM was queried, only on the surface form).
    if not systems:
        print(f"  no systems loaded, skipping {fname}")
        return
    first = next(iter(systems.values()))
    df = first["df"]
    df = df[df["item_type"] == "trap"].copy()
    if df.empty:
        print(f"  no trap rows found, skipping {fname}")
        return
    df["trap_type"] = df["trap_type"].str.split(",").str[0]

    trap_types = sorted(df["trap_type"].unique())
    data = [df[df["trap_type"] == tt][column].dropna().values for tt in trap_types]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, tick_labels=trap_types, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=14, color="black", alpha=0.5, zorder=3)

    ax.set_ylabel(title)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title} by adversarial trap type")
    is_demo = first["summary"].get("is_demo_data", False)
    _save_all_formats(fig, out_dir, fname, is_demo)


def fig_sas_distribution(systems, out_dir):
    _distribution_figure(systems, out_dir, "sas", "Sandhi Ambiguity Score (SAS)", "sas_distribution")


def fig_vdv_distribution(systems, out_dir):
    _distribution_figure(systems, out_dir, "vdv_deception", "Vibhakti Deception Score (VDV)", "vdv_distribution")


# --------------------------------------------------------------------------- #
# Figure 4 — confusion matrix, all systems as one multi-panel figure
# --------------------------------------------------------------------------- #


def fig_confusion_matrix(systems: Dict[str, Dict], out_dir: Path):
    names = list(systems.keys())
    if not names:
        print("  no systems loaded, skipping confusion_matrix")
        return
    n = len(names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    is_demo = any(systems[n_]["summary"].get("is_demo_data") for n_ in names)

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df = systems[name]["df"]
        df = df[df["item_type"].isin(["trap", "control"])]
        gt = df["ground_truth_is_hallucination_inducing"].astype(bool)
        pred = df["hybrid_flagged"].astype(bool)

        tp = int(((gt) & (pred)).sum())
        fn = int(((gt) & (~pred)).sum())
        fp = int(((~gt) & (pred)).sum())
        tn = int(((~gt) & (~pred)).sum())
        matrix = np.array([[tp, fn], [fp, tn]])

        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Flagged", "Not flagged"], fontsize=9)
        ax.set_yticklabels(["Sabotage", "Benign"], fontsize=9)
        for rr in range(2):
            for cc in range(2):
                val = matrix[rr, cc]
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(cc, rr, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
        ax.set_title(name, fontsize=11)

    # Hide any unused panels (when n doesn't fill the grid evenly)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Hybrid pipeline confusion matrix, by system", fontsize=14)
    _save_all_formats(fig, out_dir, "confusion_matrix_all_systems", is_demo)


# --------------------------------------------------------------------------- #
# Figure 5 — precision/recall vs. decision threshold, all systems overlaid
# --------------------------------------------------------------------------- #


def fig_threshold_curve(systems: Dict[str, Dict], out_dir: Path):
    names = list(systems.keys())
    if not names:
        print("  no systems loaded, skipping threshold_curve")
        return
    is_demo = any(systems[n_]["summary"].get("is_demo_data") for n_ in names)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    thresholds = np.linspace(0.0, 1.0, 51)

    for i, name in enumerate(names):
        df = systems[name]["df"]
        df = df[df["item_type"].isin(["trap", "control"])]
        gt = df["ground_truth_is_hallucination_inducing"].astype(bool).values
        scores = df["fused_probability"].values

        f1s = []
        for t in thresholds:
            pred = scores >= t
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            p = tp / (tp + fp) if (tp + fp) else np.nan
            r = tp / (tp + fn) if (tp + fn) else np.nan
            f1 = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else np.nan
            f1s.append(f1)
        ax.plot(thresholds, f1s, label=name, color=_PALETTE[i % len(_PALETTE)], linewidth=2)

    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default threshold (0.5)")
    ax.set_xlabel("Decision threshold on fused Hallucination Probability")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hybrid pipeline F1 vs. threshold, all systems")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _save_all_formats(fig, out_dir, "threshold_curve_all_systems", is_demo)


# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--out-dir", default="figures", type=Path)
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(
            f"Results directory {args.results_dir} not found. Run "
            f"scripts/run_evaluation.py first for each system you want plotted."
        )

    systems = load_results(args.results_dir)
    if not systems:
        raise SystemExit(f"No *_summary.json / *_per_example.csv pairs found in {args.results_dir}")

    print(f"Loaded {len(systems)} system(s): {list(systems.keys())}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Figure 1: bar_comparison")
    fig_bar_comparison(systems, args.out_dir)
    print("Figure 2: sas_distribution")
    fig_sas_distribution(systems, args.out_dir)
    print("Figure 3: vdv_distribution")
    fig_vdv_distribution(systems, args.out_dir)
    print("Figure 4: confusion_matrix (all systems, one file)")
    fig_confusion_matrix(systems, args.out_dir)
    print("Figure 5: threshold_curve (all systems, one file)")
    fig_threshold_curve(systems, args.out_dir)

    print(f"\nAll figures written to {args.out_dir}/ in PDF, SVG, and PNG.")
    print("Use the PDF or SVG versions in your Springer LNCS submission "
          "(vector graphics, not raster, are what typesetters want).")


if __name__ == "__main__":
    main()