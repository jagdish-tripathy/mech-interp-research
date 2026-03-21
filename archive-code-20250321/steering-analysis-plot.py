"""
steering-analysis-plot.py — Figures from Saved Raw Outputs

Reads everything from --input (raw_output dir produced by steering-analysis-run.py)
and writes all figures to --output (plots dir). No model required.

Usage:
    python steering-analysis-plot.py \
        --input  /workspace/outputs/steering_analysis/raw_output \
        --output /workspace/outputs/steering_analysis/plots

Output files
------------
  cosine_trajectory.png          Line + CI band across layers
  dual_trajectory.png            Dual-axis: cosine similarity vs info magnitude
  cosine_per_pair_heatmap.png    Heatmap: pairs × layers
  steering_4panel.png            2×2 heatmap for all 4 steering conditions
  steering_white_approve_to_deny.png   Individual heatmap per condition
  steering_black_approve_to_deny.png
  steering_white_deny_to_approve.png
  steering_black_deny_to_approve.png
  cross_layer_heatmap.png        1×2: absolute flip rates + ratio-to-baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})


# =============================================================================
# HELPERS
# =============================================================================

def _save(fig, save_dir, filename):
    p = Path(save_dir) / filename
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def load_config(input_dir):
    """Load run_config.json produced by steering-analysis-run.py."""
    path = Path(input_dir) / "run_config.json"
    with open(path) as f:
        cfg = json.load(f)
    # Ensure lists (JSON stores them correctly, but be explicit)
    cfg["layers_to_steer"] = list(cfg["layers_to_steer"])
    cfg["alphas"]           = list(cfg["alphas"])
    cfg["source_layers"]    = list(cfg["source_layers"])
    return cfg


# =============================================================================
# PART A PLOTS
# =============================================================================

def plot_cosine_trajectory(mean_cos, std_cos, label, save_dir):
    """Line + shaded CI band of mean cosine similarity across layers."""
    layers = np.arange(len(mean_cos))
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, mean_cos, color="#2563EB", linewidth=2.0, label=label, zorder=3)
    ax.fill_between(layers,
                    mean_cos - std_cos,
                    mean_cos + std_cos,
                    alpha=0.2, color="#2563EB", zorder=2)

    min_idx = int(np.argmin(mean_cos))
    ax.axvline(min_idx, color="#DC2626", linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(
        f"Min similarity\nLayer {min_idx}\n({mean_cos[min_idx]:.4f})",
        xy=(min_idx, mean_cos[min_idx]),
        xytext=(min_idx + 3, mean_cos[min_idx] - 0.002),
        fontsize=9, color="#DC2626",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1),
    )

    ax.set_xlabel("Layer (0 = embedding)", fontsize=12)
    ax.set_ylabel("Cosine Similarity (Black vs White)", fontsize=12)
    ax.set_title(
        "Residual Stream Divergence: Paired Black/White Mortgage Applications",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, len(mean_cos) - 1)
    y_min = max(0.0, mean_cos.min() - 2 * std_cos.max())
    y_max = min(1.0, mean_cos.max() + 2 * std_cos.max())
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    _save(fig, save_dir, "cosine_trajectory.png")


def plot_dual_trajectory(mean_cos, info_mag, save_dir):
    """
    Dual-axis plot: information magnitude (||v_L||, red) vs cosine
    similarity (blue dashed). Illustrates the information paradox —
    growing signal strength alongside high angular similarity.
    """
    layers = np.arange(len(mean_cos))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_mag = "#DC2626"
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Information Magnitude ||v_L|| (Euclidean Distance)",
                   color=color_mag, fontsize=12)
    ax1.plot(layers, info_mag, color=color_mag, linewidth=2.5,
             label="Signal Strength (||v_L||)")
    ax1.tick_params(axis="y", labelcolor=color_mag)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    color_cos = "#2563EB"
    ax2.set_ylabel("Cosine Similarity (Angle)", color=color_cos, fontsize=12)
    ax2.plot(layers, mean_cos, color=color_cos, linewidth=2.5, linestyle="--",
             label="Cosine Similarity")
    ax2.tick_params(axis="y", labelcolor=color_cos)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    plt.title(
        "The Information Paradox: Representational Magnitude vs. Vector Angle",
        fontsize=14
    )
    plt.tight_layout()
    _save(fig, save_dir, "dual_trajectory.png")


def plot_cosine_heatmap_per_pair(all_cosines, save_dir):
    """
    Pairs × layers heatmap. Useful for spotting whether divergence is
    uniform or driven by outliers. Subsampled to 100 pairs if larger.
    """
    data = all_cosines[:100] if all_cosines.shape[0] > 100 else all_cosines

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        data, ax=ax, cmap="RdYlBu",
        vmin=data.min(), vmax=data.max(),
        xticklabels=5, yticklabels=10,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Pair Index", fontsize=12)
    ax.set_title("Per-Pair Cosine Similarity Across Layers", fontsize=13)

    plt.tight_layout()
    _save(fig, save_dir, "cosine_per_pair_heatmap.png")


# =============================================================================
# PART B PLOTS
# =============================================================================

CONDITION_KEYS = [
    "white_approve_to_deny",
    "black_approve_to_deny",
    "white_deny_to_approve",
    "black_deny_to_approve",
]

CONDITION_TITLES = {
    "white_approve_to_deny": "White APPROVE → DENY\n(+ Black direction)",
    "black_approve_to_deny": "Black APPROVE → DENY\n(+ White direction)",
    "white_deny_to_approve":  "White DENY → APPROVE\n(+ Black direction)",
    "black_deny_to_approve":  "Black DENY → APPROVE\n(+ White direction)",
}


def plot_steering_4panel(heatmaps, n_samples, layers, alphas, save_dir):
    """
    2×2 grid showing flip-rate heatmaps for all four steering conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axis_map = {
        "white_approve_to_deny": axes[0, 0],
        "black_approve_to_deny": axes[0, 1],
        "white_deny_to_approve":  axes[1, 0],
        "black_deny_to_approve":  axes[1, 1],
    }

    for key, ax in axis_map.items():
        sns.heatmap(
            heatmaps[key], annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=1, ax=ax,
            xticklabels=[str(a) for a in alphas],
            yticklabels=[str(l) for l in layers],
            cbar_kws={"label": "Flip Rate"},
        )
        ax.set_xlabel("Steering Intensity (α)", fontsize=11)
        ax.set_ylabel("Injection Layer", fontsize=11)
        ax.set_title(
            f"{CONDITION_TITLES[key]}\n(n={n_samples[key]})",
            fontsize=12, fontweight="bold"
        )

    plt.suptitle(
        "Bidirectional Steering Analysis: All Four Conditions",
        fontsize=14, fontweight="bold", y=1.002
    )
    plt.tight_layout()
    _save(fig, save_dir, "steering_4panel.png")


def plot_steering_individual(heatmap, n_samples, layers, alphas,
                              condition_key, save_dir):
    """Single heatmap for one steering condition."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap, annot=True, fmt=".2f",
        xticklabels=[str(a) for a in alphas],
        yticklabels=[str(l) for l in layers],
        cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "Flip Rate"},
    )
    ax.set_xlabel("Steering Intensity (α)", fontsize=12)
    ax.set_ylabel("Injection Layer", fontsize=12)
    ax.set_title(
        f"Activation Steering: {CONDITION_TITLES[condition_key]}\n(n={n_samples})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, save_dir, f"steering_{condition_key}.png")


# =============================================================================
# PART C PLOTS
# =============================================================================

def plot_cross_layer_heatmap(results_df, baseline, source_layers, alphas, save_dir):
    """
    Two-panel figure:
      Left  — absolute flip rates per (source_layer, alpha)
      Right — ratio of each flip rate to the same-layer baseline
    """
    matrix = results_df.pivot(
        index="source_layer", columns="alpha", values="flip_rate"
    ).values

    baseline_array = np.array([baseline[str(a)] for a in alphas])
    ratio_matrix   = matrix / (baseline_array[None, :] + 1e-6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        matrix, annot=True, fmt=".2f",
        xticklabels=[str(a) for a in alphas],
        yticklabels=[f"L{l}" for l in source_layers],
        cmap="YlOrRd", vmin=0, vmax=1, ax=ax1,
        cbar_kws={"label": "Flip Rate"},
    )
    ax1.set_xlabel("Steering Intensity (α)", fontsize=11)
    ax1.set_ylabel("Source Layer (diff vector)", fontsize=11)
    ax1.set_title("Cross-Layer Steering: Absolute Flip Rates",
                  fontsize=12, fontweight="bold")

    sns.heatmap(
        ratio_matrix, annot=True, fmt=".2f",
        xticklabels=[str(a) for a in alphas],
        yticklabels=[f"L{l}" for l in source_layers],
        cmap="RdBu_r", center=1.0, vmin=0, vmax=2, ax=ax2,
        cbar_kws={"label": "Ratio to Baseline"},
    )
    ax2.set_xlabel("Steering Intensity (α)", fontsize=11)
    ax2.set_ylabel("Source Layer (diff vector)", fontsize=11)
    ax2.set_title("Cross-Layer Steering: Effectiveness vs Baseline",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save(fig, save_dir, "cross_layer_heatmap.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all figures from steering-analysis-run.py outputs"
    )
    parser.add_argument("--input",  type=str,
                        default="/workspace/outputs/steering_analysis/raw_output",
                        help="Directory containing raw outputs from run script")
    parser.add_argument("--output", type=str,
                        default="/workspace/outputs/steering_analysis/plots",
                        help="Directory to write figures into")
    args = parser.parse_args()

    INPUT_DIR  = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config (sweep parameters saved by run script)
    # ------------------------------------------------------------------
    cfg             = load_config(INPUT_DIR)
    label           = cfg["label"]
    LAYERS_TO_STEER = cfg["layers_to_steer"]
    ALPHAS          = cfg["alphas"]
    SOURCE_LAYERS   = cfg["source_layers"]
    TARGET_LAYER    = cfg["target_layer"]

    print(f"Config loaded: {cfg['n_pairs_geometry']} geometry pairs, "
          f"{cfg['n_pairs_steering']} steering pairs")

    # ------------------------------------------------------------------
    # Part A: Cosine trajectory figures
    # ------------------------------------------------------------------
    print("\n--- Part A: Cosine trajectory ---")
    mean_cos = np.load(INPUT_DIR / "cosine_mean.npy")
    std_cos  = np.load(INPUT_DIR / "cosine_std.npy")
    all_cos  = np.load(INPUT_DIR / "cosine_all_pairs.npy")
    info_mag = np.load(INPUT_DIR / "info_magnitude.npy")

    plot_cosine_trajectory(mean_cos, std_cos, label, OUTPUT_DIR)
    plot_dual_trajectory(mean_cos, info_mag, OUTPUT_DIR)
    plot_cosine_heatmap_per_pair(all_cos, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Part B: Steering heatmaps
    # ------------------------------------------------------------------
    print("\n--- Part B: Steering heatmaps ---")
    heatmaps = {}
    n_samples = {}

    for key in CONDITION_KEYS:
        hmap_path = INPUT_DIR / f"steering_heatmap_{key}.npy"
        det_path  = INPUT_DIR / f"steering_details_{key}.csv"

        if not hmap_path.exists():
            print(f"  WARNING: {hmap_path.name} not found, skipping {key}")
            continue

        heatmaps[key] = np.load(hmap_path)

        # Infer n_samples from details CSV if present
        if det_path.exists():
            det_df       = pd.read_csv(det_path)
            n_samples[key] = det_df["pair_id"].nunique()
        else:
            n_samples[key] = "?"

        plot_steering_individual(
            heatmaps[key], n_samples[key],
            LAYERS_TO_STEER, ALPHAS, key, OUTPUT_DIR
        )

    if len(heatmaps) == 4:
        plot_steering_4panel(heatmaps, n_samples, LAYERS_TO_STEER, ALPHAS, OUTPUT_DIR)
    else:
        print(f"  Only {len(heatmaps)}/4 conditions available; skipping 4-panel plot.")

    # ------------------------------------------------------------------
    # Part C: Cross-layer heatmap
    # ------------------------------------------------------------------
    print("\n--- Part C: Cross-layer heatmap ---")
    cross_csv      = INPUT_DIR / "cross_layer_steering.csv"
    cross_json     = INPUT_DIR / "cross_layer_baseline.json"

    if cross_csv.exists() and cross_json.exists():
        cross_results = pd.read_csv(cross_csv)
        with open(cross_json) as f:
            cross_baseline = json.load(f)
        plot_cross_layer_heatmap(
            cross_results, cross_baseline,
            SOURCE_LAYERS, ALPHAS, OUTPUT_DIR
        )
    else:
        print("  cross_layer_steering.csv or cross_layer_baseline.json not found; "
              "skipping cross-layer plot.")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")