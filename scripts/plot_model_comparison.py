"""
Generate publication-quality model comparison visualizations from eval CSVs.

Usage:
    python scripts/plot_model_comparison.py eval_results/MIP_25_old
"""

import sys
import csv
import re
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "LQ": "LQ",
    "UNet": "UNet",
    "ESRGAN": "GAN",
    "HAT": "Transformer",
    "PALETTE": "Diffusion",
    "DIP": "Zero-shot",
}
MODEL_COLORS = {
    "LQ": "#979797",
    "UNet": "#1990D4",
    "ESRGAN": "#6840D4",
    "HAT": "#77D640",
    "PALETTE": "#CC4B72",
    "DIP": "#FDB415",
}
COLOR_CYCLE = ["#1990D4", "#77D640", "#4FCAA9", "#CC4B72", "#FDB415", "#56B4E9", "#6840D4", "#979797"]
MARKER_CYCLE = ["o", "s", "D", "^", "v", "<", ">", "p"]
LINE_CYCLE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

ALL_METRICS = [
    ("psnr", "PSNR ↑", True),
    ("ssim", "SSIM ↑", True),
    ("lpips", "LPIPS ↓", False),
    ("epi", "EPI ↑", True),
    ("vessel_dice", "Vessel Dice ↑", True),
    ("density_rel_error", "Density Rel. Error ↓", False),
    ("comp_rel_error", "Comp. Rel. Error ↓", False),
    ("tortuosity_ratio", "Tortuosity Ratio → 1", None),
    ("err_vessel", "Vessel Error ↓", False),
]

BOXPLOT_METRICS = [
    ("psnr", "PSNR ↑", True),
    ("ssim", "SSIM ↑", True),
    ("lpips", "LPIPS ↓", False),
    ("epi", "EPI ↑", True),
    ("vessel_dice", "Vessel Dice ↑", True),
    ("err_vessel", "Vessel Error ↓", False),
]


def clean_model_name(raw: str) -> str:
    """Strip trailing _NN suffix and known run IDs."""
    name = re.sub(r'_\d+$', '', raw)       # e.g. UNet_50 → UNet
    name = re.sub(r'_[a-z0-9]{8,}$', '', name)  # strip trailing run ID hash
    name = re.sub(r'_zero-shot$', '', name)
    return name


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_all(csv_dir: Path):
    data = OrderedDict()
    for fpath in sorted(csv_dir.glob("*_val.csv")):
        rows = load_csv(fpath)
        raw_name = rows[0]["name"] if rows else fpath.stem
        name = clean_model_name(raw_name)
        if name not in data:
            data[name] = rows
    return data


def extract_metric(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (ValueError, KeyError):
            continue
    return np.array(vals)


def get_color(model: str, idx: int) -> str:
    return MODEL_COLORS.get(model, COLOR_CYCLE[idx % len(COLOR_CYCLE)])


def get_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def lighten_color(hex_color: str, factor: float) -> str:
    """Mix a hex color toward white; factor=1.0 keeps the color, lower = lighter."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    f = max(0.0, min(1.0, factor))
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * (1 - f)),
        int(g + (255 - g) * (1 - f)),
        int(b + (255 - b) * (1 - f)),
    )


def _ratio_sort_key(label: str) -> float:
    """Sort ratio dir labels by trailing number, e.g. MIP_50 → 50.0."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", label)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart + box plot grid
# ---------------------------------------------------------------------------

def plot_main_comparison(csv_dir: Path, save_dir: Path, data: OrderedDict):
    used_models = list(data.keys())
    if not used_models:
        print("  No model data found in", csv_dir)
        return

    n_metrics = len(BOXPLOT_METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_label, higher_is_better) in enumerate(BOXPLOT_METRICS):
        ax = axes[ax_idx]
        values = {}
        for model in used_models:
            vals = extract_metric(data[model], metric_key)
            if len(vals) > 0:
                values[model] = vals

        if not values:
            ax.set_title(f"{metric_label}\n(no data)", fontsize=9)
            ax.axis("off")
            continue

        positions = np.arange(len(used_models))
        bp = ax.boxplot(
            [values[m] for m in used_models],
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=False,
            meanline=False,
            medianprops=dict(linestyle="-", linewidth=1.5, color="gray"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(get_color(used_models[idx], idx))
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([get_label(m) for m in used_models], fontsize=8, rotation=25)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = save_dir / "model_comparison_boxplots.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Grouped bar chart with error bars
# ---------------------------------------------------------------------------

def plot_bar_chart(csv_dir: Path, save_dir: Path, data: OrderedDict):
    used_models = list(data.keys())
    if not used_models:
        print("  Skipping bar chart — no models found")
        return

    bar_metrics = [
        ("psnr", "PSNR ↑", True),
        ("ssim", "SSIM ↑", True),
        ("lpips", "LPIPS ↓", False),
        ("vessel_dice", "Vessel Dice ↑", True),
        ("epi", "EPI ↑", True),
    ]

    n_models = len(used_models)
    n_metrics = len(bar_metrics)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(n_metrics)

    for i, model in enumerate(used_models):
        means = []
        stds = []
        for metric_key, _, _ in bar_metrics:
            vals = extract_metric(data[model], metric_key)
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            stds.append(np.std(vals) if len(vals) > 0 else 0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=get_label(model), color=get_color(model, i),
               capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in bar_metrics], fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Model Comparison — Mean ± Std", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_models, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out = save_dir / "model_comparison_barchart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Metric summary table (matplotlib)
# ---------------------------------------------------------------------------

def plot_summary_table(csv_dir: Path, save_dir: Path, data: OrderedDict):
    used_models = list(data.keys())
    if not used_models:
        print("  Skipping table — no models found")
        return

    fig, ax = plt.subplots(figsize=(12, len(used_models) * 0.5 + 1.5))
    ax.axis("off")

    rows_data = []
    for model in used_models:
        row = [get_label(model)]
        for metric_key, _, _ in ALL_METRICS:
            vals = extract_metric(data[model], metric_key)
            if len(vals) > 0:
                row.append(f"{np.mean(vals):.4f}")# ± {np.std(vals):.4f}")
            else:
                row.append("—")
        rows_data.append(row)

    col_labels = ["Model"] + [m[1] for m in ALL_METRICS]
    table = ax.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F2F2F2")
        else:
            cell.set_facecolor("white")

    ax.set_title("Metric Table", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    out = save_dir / "model_comparison_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Scatter plot matrix (pairplot) — trade-off analysis
# ---------------------------------------------------------------------------

def _build_dataframe(csv_dir: Path, data: OrderedDict):
    used_models = list(data.keys())
    if not used_models:
        return {}, []
    frame = {}
    for model in used_models:
        frame[model] = {}
        for metric_key, _, _ in ALL_METRICS:
            frame[model][metric_key] = extract_metric(data[model], metric_key)
    return frame, used_models


def plot_pairplot(csv_dir: Path, save_dir: Path, data: OrderedDict):
    focus_metrics = [
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
        ("lpips", "LPIPS"),
        ("vessel_dice", "Dice"),
        ("epi", "EPI"),
    ]
    keys = [m[0] for m in focus_metrics]
    labels = [m[1] for m in focus_metrics]
    n = len(keys)

    frame, used_models = _build_dataframe(csv_dir, data)

    fig, axes = plt.subplots(n, n, figsize=(n * 2.8, n * 2.8))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if row == col:
                for i, model in enumerate(used_models):
                    vals = frame[model].get(keys[row], np.array([]))
                    if len(vals) > 0:
                        ax.hist(vals, bins=15, density=True, alpha=0.35,
                                color=get_color(model, i))
                ax.set_xlabel(labels[row], fontsize=7)
                ax.set_ylabel("Density", fontsize=7)
                ax.tick_params(labelsize=6)
            elif row < col:
                for i, model in enumerate(used_models):
                    x_vals = frame[model].get(keys[col], np.array([]))
                    y_vals = frame[model].get(keys[row], np.array([]))
                    if len(x_vals) > 0 and len(y_vals) > 0:
                        min_len = min(len(x_vals), len(y_vals))
                        ax.scatter(x_vals[:min_len], y_vals[:min_len],
                                   s=6, alpha=0.4, color=get_color(model, i),
                                   edgecolors="none")
                ax.tick_params(labelsize=6)
                ax.set_xlabel(labels[col], fontsize=7)
                ax.set_ylabel(labels[row], fontsize=7)
            else:
                ax.set_visible(False)

            ax.grid(alpha=0.15, linestyle="--")

    handles = [plt.Rectangle((0, 0), 1, 1, color=get_color(m, i), label=get_label(m))
               for i, m in enumerate(used_models)]
    fig.legend(handles=handles, loc="lower center", ncol=len(used_models),
               fontsize=8, framealpha=0.85)

    fig.suptitle("Metric Trade-off Pairplot", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = save_dir / "model_comparison_pairplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Radar chart — model profile comparison
# ---------------------------------------------------------------------------

def plot_radar(csv_dir: Path, save_dir: Path, data: OrderedDict):
    radar_metrics = [
        ("psnr", "PSNR", True),
        ("ssim", "SSIM", True),
        ("lpips", "LPIPS", False),
        ("epi", "EPI", True),
        ("vessel_dice", "Dice", True),
        ("err_vessel", "Vessel Err", False),
    ]
    keys = [m[0] for m in radar_metrics]
    labels = [m[1] for m in radar_metrics]

    frame, used_models = _build_dataframe(csv_dir, data)

    best_values = {}
    for key, _, higher_is_better in radar_metrics:
        all_vals = []
        for model in used_models:
            vals = frame[model].get(key, np.array([]))
            if len(vals) > 0:
                all_vals.extend(vals.tolist())
        if higher_is_better is True:
            best_values[key] = np.max(all_vals) if all_vals else 1.0
        elif higher_is_better is False:
            best_values[key] = np.min(all_vals) if all_vals else 0.0
        else:
            best_values[key] = 1.0

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for mi, model in enumerate(used_models):
        values = []
        for key, _, higher_is_better in radar_metrics:
            vals = frame[model].get(key, np.array([]))
            mean_val = np.mean(vals) if len(vals) > 0 else 0
            best = best_values[key]
            if higher_is_better is True:
                norm = mean_val / best if best > 0 else 0
            elif higher_is_better is False:
                norm = best / mean_val if mean_val > 0 else 0
            else:
                norm = 1.0 / (1.0 + abs(mean_val - 1.0))
            values.append(norm)
        values += values[:1]

        color = get_color(model, mi)
        marker = MARKER_CYCLE[mi % len(MARKER_CYCLE)]
        ls = LINE_CYCLE[mi % len(LINE_CYCLE)]

        ax.fill(angles, values, alpha=0.04, color=color)
        ax.plot(angles, values, linewidth=1.8, linestyle=ls,
                color=color, marker=marker, markersize=7,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8, label=get_label(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.set_title("Model Profile (1.0 = best)", fontsize=13,
                 fontweight="bold", pad=25)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12),
              fontsize=9, framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    out = save_dir / "model_comparison_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Correlation heatmap — metric redundancy check
# ---------------------------------------------------------------------------

def plot_corr_heatmap(csv_dir: Path, save_dir: Path, data: OrderedDict):
    corr_metrics = [
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
        ("lpips", "LPIPS"),
        ("epi", "EPI"),
        ("vessel_dice", "Dice"),
        ("density_rel_error", "DensErr"),
        ("comp_rel_error", "CompErr"),
        ("tortuosity_ratio", "TortRat"),
        ("err_vessel", "VesselErr"),
    ]
    keys = [m[0] for m in corr_metrics]
    labels = [m[1] for m in corr_metrics]
    n = len(keys)

    frame, _ = _build_dataframe(csv_dir, data)

    all_data = {k: [] for k in keys}
    for model in frame:
        for k in keys:
            all_data[k].extend(frame[model].get(k, []).tolist())

    corr_matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            vi = np.array(all_data[ki])
            vj = np.array(all_data[kj])
            if len(vi) > 1 and len(vj) > 1:
                corr_matrix[i, j] = np.corrcoef(vi, vj)[0, 1]
            else:
                corr_matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr_matrix[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=9)

    ax.set_title("Metric Correlation Heatmap (all models pooled)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = save_dir / "model_comparison_corr_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: Diverging dot plot
# ---------------------------------------------------------------------------

def plot_diverging_metrics(csv_dir: Path, save_dir: Path, data: OrderedDict):
    frame, used_models = _build_dataframe(csv_dir, data)

    div_metrics = [
        ("density_rel_error", "Density Rel. Error (%)", 0.0),
        ("comp_rel_error", "Comp. Rel. Error (%)", 0.0),
        ("tortuosity_ratio", "Tortuosity Ratio", 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, title, center) in zip(axes, div_metrics):
        means = []
        stds = []
        models_plot = []
        for model in reversed(used_models):
            vals = frame[model].get(key, np.array([]))
            if len(vals) > 0:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                models_plot.append(model)

        y_pos = np.arange(len(models_plot))
        ax.axvline(center, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

        for i, model in enumerate(models_plot):
            ax.errorbar(means[i], i, xerr=stds[i], fmt="o",
                        color=get_color(model, used_models.index(model)),
                        capsize=4, capthick=1.5, markersize=9,
                        markeredgecolor="white", markeredgewidth=0.8)
            vals = frame[model].get(key, np.array([]))
            if len(vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(vals, np.full_like(vals, i) + jitter, s=4, alpha=0.15,
                           color=get_color(model, used_models.index(model)))

        ax.set_yticks(y_pos)
        ax.set_yticklabels([get_label(m) for m in models_plot], fontsize=9)
        ax.set_xlabel(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")

        if title != "Tortuosity Ratio":
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.axvspan(center, max(xlim), alpha=0.04, color="red", label="over")
            ax.axvspan(min(xlim), center, alpha=0.04, color="green", label="under")

    fig.tight_layout()
    out = save_dir / "model_comparison_diverging.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 8: Sample-ratio comparison (grouped bar charts + diverging plots)
# ---------------------------------------------------------------------------

RATIO_BAR_METRICS = [
    ("psnr", "PSNR ↑", True),
    ("ssim", "SSIM ↑", True),
    ("lpips", "LPIPS ↓", False),
    ("epi", "EPI ↑", True),
    ("vessel_dice", "Vessel Dice ↑", True),
    ("err_vessel", "Vessel Error ↓", False),
]

RATIO_DIVERGING_METRICS = [
    ("density_rel_error", "Density Rel. Error (%)", 0.0),
    ("comp_rel_error", "Comp. Rel. Error (%)", 0.0),
    ("tortuosity_ratio", "Tortuosity Ratio", 1.0),
]


def plot_ratio_comparison(ratio_dirs: list, save_dir: Path):
    """Compare models across sample-ratio dirs — one plot per metric."""
    all_data = OrderedDict()
    for d in ratio_dirs:
        all_data[d.name] = load_all(d)

    ratios = sorted(all_data.keys(), key=_ratio_sort_key, reverse=True)
    print(f"  Ratios (highest → lowest): {ratios}")

    models = []
    for data in all_data.values():
        for m in data:
            if m not in models:
                models.append(m)
    print(f"  Models: {models}")

    n_ratios = len(ratios)
    shade_factors = np.linspace(1.0, 0.6, n_ratios)

    # ---- Grouped bar charts: one per metric ----
    for metric_key, metric_label, _ in RATIO_BAR_METRICS:
        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(models)), 5.5))
        x = np.arange(len(models))
        width = 0.8 / n_ratios

        for r_idx, ratio in enumerate(ratios):
            means, stds = [], []
            for model in models:
                vals = extract_metric(all_data[ratio].get(model, []), metric_key)
                means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
                stds.append(float(np.std(vals)) if len(vals) > 0 else np.nan)
            offset = (r_idx - (n_ratios - 1) / 2) * width
            colors = [lighten_color(get_color(m, models.index(m)), shade_factors[r_idx])
                      for m in models]
            ax.bar(x + offset, means, width, yerr=stds,
                   color=colors, capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([get_label(m) for m in models], fontsize=10)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} — by Sampling Ratio", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        fig.tight_layout()
        out = save_dir / f"ratio_comparison_{metric_key}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")

    # ---- Diverging dot plots: one per metric ----
    for metric_key, metric_title, center in RATIO_DIVERGING_METRICS:
        fig, ax = plt.subplots(figsize=(8.5, max(5, 0.65 * len(models))))
        y_pos = np.arange(len(models))

        for r_idx, ratio in enumerate(ratios):
            y_off = (r_idx - (n_ratios - 1) / 2) * 0.1
            for i, model in enumerate(models):
                vals = extract_metric(all_data[ratio].get(model, []), metric_key)
                if len(vals) == 0:
                    continue
                color = lighten_color(get_color(model, models.index(model)),
                                      shade_factors[r_idx])
                ax.errorbar(float(np.mean(vals)), i + y_off, xerr=float(np.std(vals)),
                            fmt="o", color=color, capsize=3, capthick=1.2,
                            markersize=7, markeredgecolor="white", markeredgewidth=0.6)
                rng = np.random.default_rng(42 + r_idx)
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(vals, np.full_like(vals, i + y_off) + jitter,
                           s=4, alpha=0.15, color=color)

        ax.axvline(center, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([get_label(m) for m in models], fontsize=10)
        ax.set_xlabel(metric_title, fontsize=10)
        ax.set_title(f"{metric_title} — by Sampling Ratio", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle="--")

        if center == 1.0:
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.axvspan(center, max(xlim), alpha=0.04, color="red")
            ax.axvspan(min(xlim), center, alpha=0.04, color="green")

        fig.tight_layout()
        out = save_dir / f"ratio_comparison_{metric_key}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_all(csv_dir_str: str = None, save_dir_str: str = None):
    args = list(sys.argv[1:]) if csv_dir_str is None else [csv_dir_str]

    # ---- Ratio comparison mode: multiple directories ----
    if len(args) > 1:
        ratio_dirs = [Path(a) for a in args]
        for d in ratio_dirs:
            if not d.exists():
                print(f"Directory not found: {d}")
                return
        save_dir = Path(save_dir_str) if save_dir_str else ratio_dirs[0].parent / "ratio_comparison"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ratio comparison mode — {len(ratio_dirs)} sample ratios: {[d.name for d in ratio_dirs]}")
        print(f"  Saving plots to: {save_dir}")
        plot_ratio_comparison(ratio_dirs, save_dir)
        print(f"\nAll ratio comparison plots saved to: {save_dir}")
        return

    csv_dir_str = args[0] if args else "eval_results/MIP_25_old"
    csv_dir = Path(csv_dir_str)
    if not csv_dir.exists():
        print(f"Directory not found: {csv_dir}")
        return

    save_dir = Path(save_dir_str) if save_dir_str else csv_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV data from: {csv_dir}")
    data = load_all(csv_dir)
    print(f"  Found models: {list(data.keys())}")

    for fn in [plot_main_comparison, plot_bar_chart, plot_summary_table,
               plot_pairplot, plot_radar, plot_corr_heatmap, plot_diverging_metrics]:
        try:
            fn(csv_dir, save_dir, data)
        except Exception as e:
            print(f"  ⚠ {fn.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    plot_all()
