import csv
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#444444',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


def plot_radial_frequency(csv_path: str):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f'[radial] CSV not found: {csv_path}')
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_keys = [k for k in rows[0].keys() if k.startswith('f')]
    freq_keys = [k for k in all_keys if all(row.get(k) not in (None, '') for row in rows)]

    data = []
    for row in rows:
        vals = np.array([float(row[k]) for k in freq_keys])
        data.append((row["model"], row["type"], vals))

    gt_profiles = [v for m, t, v in data if t == "GT"]
    if not gt_profiles:
        print('[radial] No GT profiles found')
        return
    gt_avg = np.mean(gt_profiles, axis=0)

    seen = []
    for m, t, _ in data:
        if t == "Pred" and m not in seen:
            seen.append(m)

    palette = ["#D3D3D3", "#E69F00", "#2E8CC2", "#5F39C5", "#44B698", "#C25D7B", "#B0D800", "#56B4E9", "#F0E442"]

    model_labels = {
        "LQ": "LQ",
        "UNet": "UNet",
        "ESRGAN": "GAN",
        "HAT": "Transformer",
        "PALETTE": "Diffusion",
        "DIP": "DIP",
    }

    def _fmt_label(raw):
        cleaned = re.sub(r'_\d+$', '', raw)
        for key, label in model_labels.items():
            if key.lower() in cleaned.lower():
                return label
        return cleaned

    # Skip DC (f0) which dominates — start from freq 1 for better detail
    start = 1
    freq = np.arange(start, len(gt_avg))
    gt_trim = gt_avg[start:]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                          gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08},
                                          constrained_layout=True)

    # ── Top panel: absolute radial profile ──────────────────────────────
    ax_top.plot(freq, gt_trim, color="black", linewidth=2.5, label="GT (avg)", linestyle="--", zorder=5)

    for i, model in enumerate(seen):
        preds = [v for m, t, v in data if m == model and t == "Pred"]
        if not preds:
            continue
        avg = np.mean(preds, axis=0)[start:]
        min_len = min(len(gt_trim), len(avg))
        ax_top.plot(freq[:min_len], avg[:min_len], color=palette[i % len(palette)],
                    linewidth=1.5, linestyle='-',
                    label=_fmt_label(model), zorder=3)

    ax_top.set_ylabel("Log magnitude", fontsize=11, fontweight="bold", color='#444444')
    ax_top.set_title("Radial Frequency Profile Comparison", fontsize=13, fontweight="bold", color='#444444')
    ax_top.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc', ncol=2)
    ax_top.grid(alpha=0.25, linestyle="--")
    ax_top.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # ── Bottom panel: residual (model − GT) ─────────────────────────────
    ax_bot.axhline(y=0, color='black', linewidth=0.8, linestyle='-', zorder=5)

    for i, model in enumerate(seen):
        preds = [v for m, t, v in data if m == model and t == "Pred"]
        if not preds:
            continue
        avg = np.mean(preds, axis=0)[start:]
        min_len = min(len(gt_trim), len(avg))
        residual = avg[:min_len] - gt_trim[:min_len]
        ax_bot.plot(freq[:min_len], residual, color=palette[i % len(palette)],
                    linewidth=1.5, linestyle='-',
                    label=_fmt_label(model), zorder=3)
        # Shade positive/negative regions
        ax_bot.fill_between(freq[:min_len], 0, residual,
                            color=palette[i % len(palette)], alpha=0.08)

    ax_bot.set_xlabel("Frequency radius (px)", fontsize=11, fontweight="bold", color='#444444')
    ax_bot.set_ylabel("Residual (Pred − GT)", fontsize=11, fontweight="bold", color='#444444')
    ax_bot.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc', ncol=2)
    ax_bot.grid(alpha=0.25, linestyle="--")

    # Tighter tick density on x-axis
    ax_bot.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    for spine in ['top', 'right']:
        ax_top.spines[spine].set_visible(False)
        ax_bot.spines[spine].set_visible(False)

    for ax in (ax_top, ax_bot):
        ax.tick_params(colors='#444444')

    out = csv_path.parent / "radial_frequency_comparison.png"
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[radial] Saved {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_radial_frequency.py <path/to/radial_profiles.csv>')
        sys.exit(1)
    plot_radial_frequency(sys.argv[1])
    # plot_radial_frequency("/home/v207e/GitLab/v207e/v207e/SupreRes/eval_results/MIP_25/radial_profiles.csv")  # default for testing
