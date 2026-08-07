"""
Usage:
    python eval_pipeline.py

Outputs:
    - W&B runs per checkpoint (same project, tagged "inference")
    - CSV files: eval_results/{name}_{ckpt_id}_{split}.csv
    - Console summary with average metrics
"""

from __future__ import annotations

import os
import re
import sys
import csv
import yaml
import torch
import numpy as np
import wandb
import skan
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import frangi, sobel_h, sobel_v, sobel
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label
from pathlib import Path

from pytorch_msssim import ssim
import lpips

from pipeline import SRLightningModel
from dataloader.datamodule import RSOMDataModule


# ---------------------------------------------------------------------------
# Helpers (mirror the ones used in training for consistent metrics)
# ---------------------------------------------------------------------------


_lpips_fn = None


def _get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex")
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
        _lpips_fn.eval()
    return _lpips_fn


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ssim(pred, target, data_range=1.0, size_average=True)


def _lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fn = _get_lpips_fn()
    fn = fn.to(pred.device)
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    return fn(pred, target).mean()


def _prep_for_wandb(t: torch.Tensor) -> torch.Tensor:
    return (t.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)


def extract_checkpoint_id(path: str) -> str:
    parts = Path(path).parent.name.split("_")
    return parts[-1] if parts else "unknown"


# ---------------------------------------------------------------------------
# Medical metrics helpers
# ---------------------------------------------------------------------------

def _to_grayscale(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        arr = (arr[0] + arr[1]) / 2.0
    return arr.squeeze()


def _estimate_frangi_sigmas(h: int, w: int) -> tuple:
    max_sigma = max(3, min(h, w) // 20)
    return tuple(range(1, max_sigma + 2, 3))


def _threshold_vesselness(vesselness: np.ndarray, img_size: int) -> np.ndarray:
    flat = vesselness[vesselness > 0]
    if len(flat) == 0:
        return np.zeros_like(vesselness, dtype=bool)
    thresh = np.percentile(flat, 85)
    thresh = max(thresh, 0.002)
    mask = vesselness > thresh
    min_size = max(5, img_size // 50)
    return remove_small_objects(mask, min_size=min_size)


def _get_vessel_mask(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape
    sigmas = _estimate_frangi_sigmas(h, w)
    vesselness = frangi(img_np, sigmas=sigmas)
    return _threshold_vesselness(vesselness, min(h, w))


def _epi(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    mag_p = np.sqrt(sobel_h(pred_np) ** 2 + sobel_v(pred_np) ** 2)
    mag_t = np.sqrt(sobel_h(target_np) ** 2 + sobel_v(target_np) ** 2)
    corr = np.corrcoef(mag_p.ravel(), mag_t.ravel())[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Vessel metrics: Dice, density error, component error, tortuosity (skan)
# ---------------------------------------------------------------------------

def _skeleton_tortuosity(skel: np.ndarray) -> dict:
    if not skel.any():
        return {'tortuosity': 0.0, 'tortuosity_std': 0.0, 'n_paths': 0}
    s = skan.Skeleton(skel)
    n = s.n_paths
    if n == 0:
        return {'tortuosity': 0.0, 'tortuosity_std': 0.0, 'n_paths': 0}
    arcs = np.array(s.path_lengths(), dtype=np.float64)
    chords = np.zeros(n, dtype=np.float64)
    for i in range(n):
        coords = s.path_coordinates(i)
        if len(coords) < 2:
            continue
        start, end = coords[0], coords[-1]
        chords[i] = np.sqrt(float((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
    valid = (chords > 0.5) & (arcs >= 5)
    if not valid.any():
        return {'tortuosity': 0.0, 'tortuosity_std': 0.0, 'n_paths': 0}
    per_path = arcs[valid] / chords[valid]
    n_valid = int(valid.sum())
    return {
        'tortuosity': float(arcs[valid].sum() / chords[valid].sum()),
        'tortuosity_std': float(per_path.std()),
        'n_paths': n_valid,
    }


def _vessel_metrics(pred_np: np.ndarray, target_np: np.ndarray) -> dict:
    mask_pred = _get_vessel_mask(pred_np)
    mask_target = _get_vessel_mask(target_np)

    inter = np.sum(mask_pred & mask_target)
    sum_p = float(np.sum(mask_pred))
    sum_t = float(np.sum(mask_target))
    dice = 2.0 * inter / (sum_p + sum_t) if (sum_p + sum_t) > 0 else 0.0

    density_error = float(np.mean(mask_pred)) - float(np.mean(mask_target))
    density_rel_error = density_error / (float(np.mean(mask_target)) + 1e-8) * 100.0

    labeled_pred = label(mask_pred)
    labeled_target = label(mask_target)
    n_pred = int(labeled_pred.max())
    n_target = int(labeled_target.max())
    comp_rel_error = (n_pred - n_target) / max(n_target, 1) * 100.0

    skel_pred = skeletonize(mask_pred)
    skel_target = skeletonize(mask_target)
    tort_gt = _skeleton_tortuosity(skel_target)
    tort_pred = _skeleton_tortuosity(skel_pred)

    tort_ratio = tort_pred['tortuosity'] / tort_gt['tortuosity'] if tort_gt['tortuosity'] > 0 else 1.0

    err_vessel = float(np.mean(np.abs(pred_np - target_np)[mask_target])) if mask_target.any() else 0.0

    return {
        "vessel_dice": dice,
        "density_rel_error": density_rel_error,
        "comp_rel_error": comp_rel_error,
        "tortuosity_gt": tort_gt['tortuosity'],
        "tortuosity_pred": tort_pred['tortuosity'],
        "tortuosity_ratio": tort_ratio,
        "err_vessel": err_vessel,
    }


# ---------------------------------------------------------------------------
# Vessel overlay visualization
# ---------------------------------------------------------------------------


def _render_contour_view(img_np, masks_and_styles, title):
    """Render grayscale image + contour overlays -> numpy array (H, W, 3)."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    for mask, color, ls in masks_and_styles:
        ax.contour(mask, levels=[0.5], colors=color, linewidths=0.7, linestyles=ls)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


def _make_vessel_overlay(
    target_np: np.ndarray,
    pred_np: np.ndarray,
    target_mask: np.ndarray,
    pred_mask: np.ndarray,
    model_name: str,
    sample_idx: int,
    save_dir: Path,
):
    """Create 3-panel figure (GT | Pred | Overlap) saved locally + return 3 W&B views."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, img, mask, title, color in [
        (axes[0], target_np, target_mask, f"GT ({model_name})", "#0072B2"),
        (axes[1], pred_np, pred_mask, f"Pred ({model_name})", "#D55E00"),
    ]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.contour(mask, levels=[0.5], colors=color, linewidths=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    axes[2].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(target_mask, levels=[0.5], colors="#0072B2", linewidths=0.7, linestyles="-")
    axes[2].contour(pred_mask, levels=[0.5], colors="#D55E00", linewidths=0.7, linestyles="--")
    axes[2].set_title(f"Overlap ({model_name})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    fig.tight_layout(pad=0.3)
    path = save_dir / f"vessel_viz_{model_name}_sample{sample_idx:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gt_viz = _render_contour_view(target_np, [(target_mask, "#0072B2", "-")], f"GT ({model_name})")
    pred_viz = _render_contour_view(pred_np, [(pred_mask, "#D55E00", "-")], f"Pred ({model_name})")
    overlap_viz = _render_contour_view(
        target_np,
        [(target_mask, "#0072B2", "-"), (pred_mask, "#D55E00", "--")],
        f"Overlap ({model_name})",
    )

    return path, gt_viz, pred_viz, overlap_viz


# ---------------------------------------------------------------------------
# Error map + frequency analysis visualizations
# ---------------------------------------------------------------------------


def _make_error_map(
    target_np: np.ndarray,
    pred_np: np.ndarray,
    model_name: str,
    sample_idx: int,
    save_dir: Path,
):
    """Absolute error heat map PNG + numpy array for W&B."""
    error = np.abs(pred_np - target_np)
    vmax = error.max() if error.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(error, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"Absolute Error ({model_name})", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.3)
    path = save_dir / f"error_map_{model_name}_sample{sample_idx:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return path, arr


def _make_fft_comparison(
    target_np: np.ndarray,
    pred_np: np.ndarray,
    model_name: str,
    sample_idx: int,
    save_dir: Path,
):
    """2-panel figure: GT log-magnitude FFT | Pred log-magnitude FFT."""
    def log_spec(img):
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-10)

    gt_spec = log_spec(target_np)
    pred_spec = log_spec(pred_np)
    vmin = min(gt_spec.min(), pred_spec.min())
    vmax = max(gt_spec.max(), pred_spec.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt_spec, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"GT FFT ({model_name})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(pred_spec, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Pred FFT ({model_name})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout(pad=0.3)
    path = save_dir / f"fft_{model_name}_sample{sample_idx:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return path, arr


RADIAL_PROFILES_CSV = "radial_profiles.csv"


def _compute_radial_profile(img_np: np.ndarray) -> np.ndarray:
    """Compute radial average of 2D log-magnitude FFT spectrum."""
    fft = np.fft.fft2(img_np)
    spec = np.log(np.abs(np.fft.fftshift(fft)) + 1e-10)
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    profile = np.zeros(max_r)
    for i in range(max_r):
        mask = (r >= i) & (r < i + 1)
        profile[i] = spec[mask].mean()
    return profile


def _append_radial_profile(csv_path: Path, model_name: str, profile: np.ndarray, profile_type: str):
    """Append one radial profile row to CSV. type='GT' or 'Pred'."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model", "type"] + [f"f{i}" for i in range(len(profile))]
            writer.writerow(header)
        writer.writerow([model_name, profile_type] + [f"{v:.6f}" for v in profile])


def _plot_radial_from_csv(csv_path: Path, save_dir: Path):
    """Read radial_profiles.csv and plot 2-panel figure: absolute + residual."""
    if not csv_path.exists():
        return

    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_keys = [k for k in rows[0].keys() if k.startswith('f')]
    freq_keys = [k for k in all_keys if all(row.get(k) not in (None, '') for row in rows)]

    for row in rows:
        vals = np.array([float(row[k]) for k in freq_keys])
        data.append((row["model"], row["type"], vals))

    gt_profiles = [v for m, t, v in data if t == "GT"]
    if not gt_profiles:
        return
    gt_avg = np.mean(gt_profiles, axis=0)

    seen = []
    for m, t, _ in data:
        if t == "Pred" and m not in seen:
            seen.append(m)

    palette = ["#CF7104", "#0072B2", "#009E73", "#CC79A7", "#5603C4", "#E69F00", "#56B4E9", "#F0E442"]

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

    start = 1
    freq = np.arange(start, len(gt_avg))
    gt_trim = gt_avg[start:]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                          gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08},
                                          constrained_layout=True)

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
        ax_bot.fill_between(freq[:min_len], 0, residual,
                            color=palette[i % len(palette)], alpha=0.08)

    ax_bot.set_xlabel("Frequency radius (px)", fontsize=11, fontweight="bold", color='#444444')
    ax_bot.set_ylabel("Residual (Pred − GT)", fontsize=11, fontweight="bold", color='#444444')
    ax_bot.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc', ncol=2)
    ax_bot.grid(alpha=0.25, linestyle="--")

    ax_bot.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    for spine in ['top', 'right']:
        ax_top.spines[spine].set_visible(False)
        ax_bot.spines[spine].set_visible(False)

    for ax in (ax_top, ax_bot):
        ax.tick_params(colors='#444444')

    fig.tight_layout(pad=0.5)
    out = save_dir / "radial_frequency_comparison.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[radial] Saved {out}")


# ---------------------------------------------------------------------------
# Edge comparison visualization (grayscale original + sobel edge side-by-side)
# ---------------------------------------------------------------------------


def _make_edge_comparison(
    target_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    target_np: np.ndarray,
    pred_np: np.ndarray,
    model_name: str,
    sample_idx: int,
    save_dir: Path,
):
    """4-panel figure: GT (RGB) | GT Edges | Pred (RGB) | Pred Edges."""

    gt_edge = sobel(target_np)
    pred_edge = sobel(pred_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(target_rgb, vmin=0, vmax=1)
    axes[1].imshow(gt_edge, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(pred_rgb, vmin=0, vmax=1)
    axes[3].imshow(pred_edge, cmap="gray", vmin=0, vmax=1)

    for ax, title in [
        (axes[0], f"GT ({model_name})"),
        (axes[1], f"GT Edges ({model_name})"),
        (axes[2], f"Pred ({model_name})"),
        (axes[3], f"Pred Edges ({model_name})"),
    ]:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    path = save_dir / f"edge_gt_pred_{model_name}_sample{sample_idx:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared evaluation logic for one split
# ---------------------------------------------------------------------------

METRICS_HEADER = [
    "sample", "name",
    "psnr", "ssim", "lpips",
    "epi",
    "vessel_dice", "density_rel_error", "comp_rel_error",
    "tortuosity_gt", "tortuosity_pred", "tortuosity_ratio",
    "err_vessel",
]

METRIC_KEYS = [
    "psnr", "ssim", "lpips", "epi",
    "vessel_dice", "density_rel_error", "comp_rel_error",
    "tortuosity_gt", "tortuosity_pred", "tortuosity_ratio",
    "err_vessel",
]


def _avg(vals):
    return float(np.mean(vals)) if vals else 0.0


def _compute_per_sample(x_i, y_i, device, mode):
    """Compute all metrics between a single prediction/input and target."""
    batch_singleton = x_i.unsqueeze(0)
    y_singleton = y_i.unsqueeze(0)

    psnr_val = _psnr(batch_singleton, y_singleton).item()
    ssim_val = _ssim(batch_singleton, y_singleton).item()
    lpips_val = _lpips(batch_singleton, y_singleton).item()

    pred_np = _to_grayscale(batch_singleton[0])
    target_np = _to_grayscale(y_singleton[0])

    epi_val = _epi(pred_np, target_np)
    vm = _vessel_metrics(pred_np, target_np)

    metrics = {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "epi": epi_val,
        **vm,
    }
    return metrics


def _evaluate_split(loader, csv_path, get_pred_fn, model_name, split, global_step, run, viz_dir=None, device=None, use_slices=False, log_scale_C=0.0):
    """Evaluate one data split and write CSV + W&B logs."""
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(METRICS_HEADER)

        batch_metrics = {k: [] for k in METRIC_KEYS}
        sample_counter = 0

        for batch_idx, batch in enumerate(loader):
            if use_slices:
                # ---- Slice mode: batch is (lq_vol, hq_vol, hq_mip_gt) ----
                lq_vol, hq_vol, hq_mip_gt = batch
                lq_vol = lq_vol.squeeze(0).float().to(device)   # (Y, 3, H, W)
                hq_vol = hq_vol.squeeze(0).float().to(device)
                hq_mip_gt = hq_mip_gt.squeeze(0).float().to(device)
                Y = lq_vol.shape[0]

                pred_frames = []
                if "dip" in model_name.lower():
                    for s in range(Y):
                        pred_frames.append(get_pred_fn(lq_vol[s].unsqueeze(0)).squeeze(0))
                    pred_vol = torch.stack(pred_frames)  # (Y, 3, H, W)
                else:
                    bs_val = min(8, Y)
                    for s in range(0,Y, bs_val):
                        pred_frames.append(get_pred_fn(lq_vol[s:s+bs_val]))
                    pred_vol = torch.cat(pred_frames)  # (Y, 3, H, W)

                # Undo log scale
                if log_scale_C:
                    pred_vol = torch.expm1(pred_vol) / log_scale_C
                    hq_vol   = torch.expm1(hq_vol) / log_scale_C
                    lq_vol   = torch.expm1(lq_vol) / log_scale_C
                    pred_vol = pred_vol.clamp(0, 1)

                # MIP for metrics + visualization
                y_hat_mip = pred_vol.max(dim=0)[0]   # (3, H, W)
                x_mip = lq_vol.max(dim=0)[0]          # LQ MIP for visualization

                # Per-sample metrics (MIP only — comparable to MIP-trained models)
                with torch.no_grad():
                    m = _compute_per_sample(y_hat_mip, hq_mip_gt, device, "mip")

                # Set up x_i, y_i, pred_i for visualization (use MIPs)
                x_i = x_mip
                y_i = hq_mip_gt
                pred_i = y_hat_mip

                log_dict = {f"inference/{split}/{k}": m[k] for k in METRIC_KEYS}

                # Per-slice metrics (W&B only, not CSV)
                slice_metrics = []
                log_dict[f"inference/{split}/slice_samples"] = Y
                for s in range(min(Y, 20)):  # sample up to 20 slices
                    sm = _compute_per_sample(pred_vol[s], hq_vol[s], device, "slice")
                    slice_metrics.append(sm)
                if slice_metrics:
                    for k in METRIC_KEYS:
                        vals = [sm[k] for sm in slice_metrics]
                        log_dict[f"inference/{split}/slice_avg_{k}"] = _avg(vals)

                # Log slice images (random subset)
                if sample_counter % 10 == 0:
                    rng = np.random.default_rng()
                    for s in rng.choice(Y, min(4, Y), replace=False):
                        log_dict[f"inference/{split}/slice_{s}"] = [
                            wandb.Image(_prep_for_wandb(lq_vol[s]), caption=f"LQ Slice {s} ({model_name})"),
                            wandb.Image(_prep_for_wandb(hq_vol[s]), caption=f"HQ Slice {s} ({model_name})"),
                            wandb.Image(_prep_for_wandb(pred_vol[s]), caption=f"Pred Slice {s} ({model_name})"),
                        ]

                # batch_size=1 for slice mode (full volume)
                batch_size = 1
                # Will loop once (batch_size=1) below with x_i, y_i, pred_i as MIPs
            else:
                # ---- Existing MIP mode (completely unchanged) ----
                x, y = batch
                x, y = x.float().to(device), y.float().to(device)
                batch_size = x.shape[0]

                preds = get_pred_fn(x)
                log_dict = {}

            for b in range(batch_size):
                if not use_slices:
                    x_i = x[b]
                    y_i = y[b]
                    pred_i = preds[b]

                    with torch.no_grad():
                        m = _compute_per_sample(pred_i, y_i, device, "model")

                    for k in METRIC_KEYS:
                        batch_metrics[k].append(m[k])

                    log_dict = {f"inference/{split}/{k}": m[k] for k in METRIC_KEYS}

                else:
                    # Slice mode: store MIP metrics in batch_metrics for summary
                    for k in METRIC_KEYS:
                        batch_metrics[k].append(m[k])

                if sample_counter % 10 == 0:
                    log_dict[f"inference/{split}/visual_comparison"] = [
                        wandb.Image(_prep_for_wandb(x_i),   caption=f"LQ Input ({model_name})"),
                        wandb.Image(_prep_for_wandb(y_i),   caption=f"HQ Ground Truth ({model_name})"),
                        wandb.Image(_prep_for_wandb(pred_i), caption=f"Prediction ({model_name})"),
                    ]
                    if viz_dir is not None:
                        pred_np = _to_grayscale(pred_i)
                        target_np = _to_grayscale(y_i)
                        pred_mask = _get_vessel_mask(pred_np)
                        target_mask = _get_vessel_mask(target_np)
                        viz_path, gt_viz, pred_viz, overlap_viz = _make_vessel_overlay(
                            target_np, pred_np, target_mask, pred_mask,
                            model_name, sample_counter, viz_dir,
                        )
                        log_dict[f"inference/{split}/vessel_overlap"] = wandb.Image(overlap_viz, caption=f"Vessel Overlap ({model_name})")
                        target_rgb = y_i.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
                        pred_rgb = pred_i.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
                        _make_edge_comparison(
                            target_rgb, pred_rgb, target_np, pred_np,
                            model_name, sample_counter, viz_dir,
                        )
                        err_path, err_viz = _make_error_map(
                            target_np, pred_np, model_name, sample_counter, viz_dir,
                        )
                        log_dict[f"inference/{split}/error_map"] = wandb.Image(err_viz, caption=f"Error Map ({model_name})")
                        fft_path, fft_viz = _make_fft_comparison(
                            target_np, pred_np, model_name, sample_counter, viz_dir,
                        )
                        log_dict[f"inference/{split}/fft_comparison"] = wandb.Image(fft_viz, caption=f"FFT ({model_name})")
                        rad_csv = csv_path.parent / RADIAL_PROFILES_CSV
                        _append_radial_profile(rad_csv, model_name, _compute_radial_profile(target_np), "GT")
                        _append_radial_profile(rad_csv, model_name, _compute_radial_profile(pred_np), "Pred")

                wandb.log(log_dict, step=global_step)
                global_step += 1

                csv_writer.writerow([
                    sample_counter, model_name,
                    f"{m['psnr']:.4f}",
                    f"{m['ssim']:.4f}",
                    f"{m['lpips']:.6f}",
                    f"{m['epi']:.4f}",
                    f"{m['vessel_dice']:.4f}",
                    f"{m['density_rel_error']:.2f}",
                    f"{m['comp_rel_error']:.2f}",
                    f"{m['tortuosity_gt']:.4f}",
                    f"{m['tortuosity_pred']:.4f}",
                    f"{m['tortuosity_ratio']:.4f}",
                    f"{m['err_vessel']:.6f}",
                ])
                sample_counter += 1

    avg = {k: _avg(batch_metrics[k]) for k in METRIC_KEYS}
    wandb.run.summary.update({f"inference/{split}/avg_{k}": avg[k] for k in METRIC_KEYS})

    print(f"  [{split}]  PSNR: {avg['psnr']:.4f}  SSIM: {avg['ssim']:.4f}  LPIPS: {avg['lpips']:.6f}")
    print(f"           EPI: {avg['epi']:.4f}  Dice: {avg['vessel_dice']:.4f}"
          f"  DensErr: {avg['density_rel_error']:.1f}%  CompErr: {avg['comp_rel_error']:.1f}%")
    print(f"           Tort_GT: {avg['tortuosity_gt']:.4f}  Tort_Pred: {avg['tortuosity_pred']:.4f}"
          f"  Tort_Ratio: {avg['tortuosity_ratio']:.4f}  VesselErr: {avg['err_vessel']:.6f}")
    print(f"  [{split}]  CSV saved: {csv_path}")

    return global_step


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_entry: dict, cfg: dict, device: torch.device, csv_dir: Path):
    name = model_entry["name"]
    use_slices = cfg["data"].get("use_slices", False)
    use_log_scale = cfg["data"].get("use_log_scale", False)
    log_scale_C = cfg["data"].get("log_scale_factor", 0.0)

    if model_entry.get("type") == "dip":
        # DIP: zero-shot, no checkpoint needed
        print(f"\n{'=' * 70}")
        print(f"[eval] DIP (zero-shot): {name}")
        model = SRLightningModel(
            model_type="DIP",
            lr=model_entry.get("lr", 0.01),
            out_channels=3,
            loss_type="l1",
            input_depth=model_entry.get("input_depth", 32),
            n_scales=model_entry.get("n_scales", 5),
            need_sigmoid=True,
            use_slices=use_slices,
            use_log_scale=use_log_scale,
            log_scale_factor=log_scale_C,
            dip_num_iterations=model_entry.get("num_iterations", 2400),
            dip_tv_weight=model_entry.get("tv_weight", 0.0),
            dip_input_noise=model_entry.get("input_noise", True),
            dip_reg_noise_std=model_entry.get("reg_noise_std", 0.033),
            dip_ema_weight=model_entry.get("ema_weight", 0.99),
            dip_backtrack_thresh=model_entry.get("backtrack_thresh", 1.05),
            dip_num_runs=model_entry.get("num_runs", 1),
        )
        model = model.to(device)
        model.eval()
        model_type = "DIP"
        ckpt_id = "zero-shot"
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        # Regular model: load from checkpoint
        ckpt_path = str(Path(model_entry["checkpoint"]).resolve())
        print(f"\n{'=' * 70}")
        print(f"[eval] Loading model: {name}  |  checkpoint: {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            print(f"  ⚠ checkpoint not found — skipping")
            return
        try:
            model = SRLightningModel.load_from_checkpoint(ckpt_path, strict=False)
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            return
        model = model.to(device)
        model.eval()
        model_type = model.model_type
        ckpt_id = extract_checkpoint_id(ckpt_path)
        print(f"  Model type: {model_type}  |  Params: {sum(p.numel() for p in model.parameters()):,}")
        # Read slice flags from model checkpoint
        use_slices = getattr(model, 'use_slices', False)
        log_scale_C = getattr(model, 'log_scale_C', 0.0)
        if use_slices:
            print(f"  Slice mode: True  |  log_scale_C: {log_scale_C}")

    # ... (no changes to DIP path above)

    tags = cfg.get("wandb", {}).get("tags", []) + [model_type, ckpt_id, f"ckpt_{name}"]
    run = wandb.init(
        project=cfg.get("wandb", {}).get("project", "image-sr"),
        name=f"{name}_inference_{ckpt_id}",
        tags=tags,
        reinit=True,
    )

    global_step = 0
    for split in cfg.get("splits", ["val"]):
        print(f"\n  [{split}] Evaluating...")

        dm = RSOMDataModule(
            folder_path=cfg["data"]["folder_path"],
            batch_size=cfg["data"].get("batch_size", 1),
            num_workers=cfg["data"].get("num_workers", 0),
            pre_split=cfg["data"].get("pre_split", True),
            patch_size=128,
            use_slices=use_slices,
            use_log_scale=cfg["data"].get("use_log_scale", False),
            log_scale_factor=cfg["data"].get("log_scale_factor", 0.0),
            n_slices=cfg["data"].get("n_slices", 168),
        )
        lightning_stage = "validate" if split == "val" else "test"
        dm.setup(stage=lightning_stage)
        loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

        csv_path = csv_dir / f"{name}_{ckpt_id}_{split}.csv"

        def get_pred(x):
            with torch.no_grad():
                return model(x)

        global_step = _evaluate_split(loader, csv_path, get_pred, name, split, global_step, run, viz_dir=csv_dir, device=device, use_slices=use_slices, log_scale_C=log_scale_C)

    run.finish()


# ---------------------------------------------------------------------------
# LQ evaluation (LQ vs HQ — no model)
# ---------------------------------------------------------------------------

def evaluate_lq(cfg: dict, device: torch.device, csv_dir: Path):
    print(f"\n{'=' * 70}")
    print("[eval] Computing lq (LQ vs HQ — no model)...")

    use_slices = cfg["data"].get("use_slices", False)
    log_scale_C = cfg["data"].get("log_scale_factor", 0.0)

    tags = cfg.get("wandb", {}).get("tags", []) + ["LQ"]
    run = wandb.init(
        project=cfg.get("wandb", {}).get("project", "image-sr"),
        name="LQ_inference",
        tags=tags,
        reinit=True,
    )

    global_step = 0
    for split in cfg.get("splits", ["val"]):
        print(f"\n  [{split}] LQ...")

        dm = RSOMDataModule(
            folder_path=cfg["data"]["folder_path"],
            batch_size=cfg["data"].get("batch_size", 1),
            num_workers=cfg["data"].get("num_workers", 0),
            pre_split=cfg["data"].get("pre_split", True),
            patch_size=128,
            use_slices=use_slices,
            use_log_scale=cfg["data"].get("use_log_scale", False),
            log_scale_factor=cfg["data"].get("log_scale_factor", 0.0),
            n_slices=cfg["data"].get("n_slices", 168),
        )
        lightning_stage = "validate" if split == "val" else "test"
        dm.setup(stage=lightning_stage)
        loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

        csv_path = csv_dir / f"LQ_{split}.csv"

        def get_pred(x):
            return x  # LQ: no model, LQ directly compared to HQ

        global_step = _evaluate_split(loader, csv_path, get_pred, "LQ", split, global_step, run, viz_dir=csv_dir, device=device, use_slices=use_slices, log_scale_C=log_scale_C)

    run.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = Path(__file__).parent / "configs" / "eval_config.yaml"
    if not config_path.exists():
        print(f"[eval] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(__file__).parent.resolve()
    csv_dir = project_root / cfg.get("output", {}).get("csv_dir", "eval_results")
    csv_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    if not cfg.get("skip_lq", False):
        evaluate_lq(cfg, device, csv_dir)

    for model_entry in cfg.get("models", []):
        try:
            evaluate_model(model_entry, cfg, device, csv_dir)
        except Exception as e:
            print(f"  ⚠ Model evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 70}")
    print("[eval] All models evaluated.")

    print(f"\n{'=' * 70}")
    print("[eval] Generating box plots...")
    try:
        from scripts.plot_model_comparison import plot_all
        plot_all(str(csv_dir))
    except Exception as e:
        print(f"  ⚠ Box plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("[eval] Generating combined radial frequency plot...")
    try:
        _plot_radial_from_csv(csv_dir / RADIAL_PROFILES_CSV, csv_dir)
    except Exception as e:
        print(f"  ⚠ Radial frequency plot failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
