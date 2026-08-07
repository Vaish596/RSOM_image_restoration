"""
patch_utils.py
==============
Utility functions for paired patch extraction from RSOM HQ/LQ image pairs.

Design principles:
  - All crop operations are PAIRED: identical spatial coordinates applied to
    both HQ and LQ, preserving the ground-truth correspondence.
  - Patch rejection prevents empty / signal-free patches from polluting
    the training set (common at image borders in photoacoustic data).
  - Functions are pure (no side effects) and stateless for easy testing.

Coordinate convention:
  Images are stored as (H, W) or (H, W, C).
  Patch coordinates are (row_start, col_start) → (row_end, col_end).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hw(image: np.ndarray) -> Tuple[int, int]:
    """Return (H, W) regardless of whether image is (H,W) or (H,W,C)."""
    return image.shape[0], image.shape[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_pad_image(
    image: np.ndarray,
    min_size: int,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Pad image symmetrically so that both spatial dimensions are >= min_size.

    Used as a safety net before patch extraction: if the stored slice is
    slightly smaller than the requested patch_size, padding avoids a crash
    while keeping the intensity distribution reasonable (reflect is preferred
    over zero-padding because zeros look like background in photoacoustic data).

    Args:
        image:    (H, W) or (H, W, C) float32 array.
        min_size: Required minimum for H and W.
        mode:     np.pad mode; 'reflect' or 'constant'.

    Returns:
        Padded array of the same dtype.
    """
    h, w = _hw(image)
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)

    if pad_h == 0 and pad_w == 0:
        return image

    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    if image.ndim == 2:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    return np.pad(image, padding, mode=mode)


def is_informative_patch(
    patch: np.ndarray,
    threshold: float = 0.02,
) -> bool:
    """
    Return True when a patch carries meaningful photoacoustic signal.

    Rationale:
      RSOM images have large low-signal border regions after cropping.
      Training on entirely dark patches wastes capacity and biases the
      model towards predicting zero.  We keep a patch only when its mean
      intensity exceeds a small threshold (default 2 % of [0, 1] range).

    Args:
        patch:     Numpy array of any shape, values expected in [0, 1].
        threshold: Minimum mean intensity to accept the patch.

    Returns:
        bool
    """
    return float(patch.mean()) > threshold


def paired_random_crop(
    hq: np.ndarray,
    lq: np.ndarray,
    patch_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a random crop of size (patch_size × patch_size) from an HQ/LQ pair.

    IMPORTANT: Both images receive the EXACT SAME (top, left) offset, which is
    mandatory for inverse-problem restoration training where pixel-level spatial
    correspondence must be preserved.

    Args:
        hq:         (H, W) or (H, W, C) float32 HQ array.
        lq:         Same shape as hq.
        patch_size: Side length of the square crop.
        rng:        Optional numpy Generator for reproducibility.  If None a
                    default Generator is created (non-reproducible).

    Returns:
        (hq_patch, lq_patch) both of shape (patch_size, patch_size[, C]).

    Raises:
        ValueError: If patch_size exceeds either spatial dimension.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = _hw(hq)

    if patch_size > h or patch_size > w:
        raise ValueError(
            f"patch_size={patch_size} is larger than image size ({h}, {w}). "
            "Either reduce patch_size or enable maybe_pad_image."
        )

    # Sample a single (top, left) origin — applied identically to HQ and LQ
    top  = int(rng.integers(0, h - patch_size + 1))
    left = int(rng.integers(0, w - patch_size + 1))

    if hq.ndim == 2:
        hq_patch = hq[top:top + patch_size, left:left + patch_size]
        lq_patch = lq[top:top + patch_size, left:left + patch_size]
    else:
        hq_patch = hq[top:top + patch_size, left:left + patch_size, :]
        lq_patch = lq[top:top + patch_size, left:left + patch_size, :]

    return hq_patch, lq_patch


def sample_informative_patch(
    hq: np.ndarray,
    lq: np.ndarray,
    patch_size: int,
    threshold: float = 0.02,
    max_tries: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repeatedly sample random patches until an informative one is found.

    Strategy:
      Try up to max_tries times to find a patch where the HQ mean > threshold.
      We use HQ (not LQ) as the reference because LQ may have suppressed signal
      due to undersampling — using LQ would unfairly reject valid training pairs.
      If no informative patch is found within max_tries, the last sample is
      returned to avoid an infinite loop.

    Args:
        hq:         (H, W[, C]) float32 array.
        lq:         Same spatial shape as hq.
        patch_size: Side of the square crop.
        threshold:  Minimum mean intensity for acceptance.
        max_tries:  Maximum sampling attempts before giving up.
        rng:        Optional numpy Generator.

    Returns:
        (hq_patch, lq_patch)
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries):
        hq_patch, lq_patch = paired_random_crop(hq, lq, patch_size, rng)
        if is_informative_patch(hq_patch, threshold):
            return hq_patch, lq_patch

    # Return the last sampled patch even if uninformative (fallback)
    return hq_patch, lq_patch  # type: ignore[return-value]


def validate_pair_shapes(hq: np.ndarray, lq: np.ndarray, sample_id: str = "") -> None:
    """
    Assert that HQ and LQ arrays have matching spatial dimensions.

    This is the first sanity check run at dataset load time.  A shape mismatch
    almost certainly indicates a bug in the dataset creation pipeline (e.g. a
    crop applied only to one channel).

    Args:
        hq:        HQ array.
        lq:        LQ array.
        sample_id: Optional sample identifier for clearer error messages.

    Raises:
        ValueError on mismatch.
    """
    if hq.shape != lq.shape:
        prefix = f"[{sample_id}] " if sample_id else ""
        raise ValueError(
            f"{prefix}Shape mismatch: HQ={hq.shape} vs LQ={lq.shape}. "
            "Check the dataset creation pipeline for asymmetric processing."
        )


def validate_patch_size(patch_size: int, hq: np.ndarray, sample_id: str = "") -> None:
    """
    Assert that patch_size is smaller than the image's spatial dimensions.

    Args:
        patch_size: Requested patch size.
        hq:         HQ (or LQ) image array to check against.
        sample_id:  Optional label for error messages.

    Raises:
        ValueError when patch_size exceeds the image.
    """
    h, w = _hw(hq)
    if patch_size > h or patch_size > w:
        prefix = f"[{sample_id}] " if sample_id else ""
        raise ValueError(
            f"{prefix}patch_size={patch_size} exceeds image size (H={h}, W={w}). "
            "Reduce patch_size or call maybe_pad_image before extraction."
        )


def sliding_window_predict(model, lq: torch.Tensor,
                            patch_size: int = 128,
                            stride: int = 64,
                            return_patches: bool = False,
                            hq: torch.Tensor | None = None):
    """
    Run model over a full image using overlapping patches, then stitch
    with overlap averaging to avoid hard boundary seams.

    Args:
        model:      Your restoration network (eval mode, on correct device).
        lq:         (C, H, W) input tensor — a single image, no batch dim.
        patch_size: Spatial size of patches the model expects.
        stride:     Step between patches. stride < patch_size = overlap.
        return_patches: If True, also return individual (lq, hq, pred) patches.
        hq:         (C, H, W) ground truth — required when return_patches=True.

    Returns:
        If return_patches=False: (C, H, W) stitched prediction tensor.
        If return_patches=True:  (stitched_pred, list_of_patches) where each
            patch is (lq_patch, hq_patch, pred_patch, top, left).
    """
    device = lq.device
    C, H, W = lq.shape

    if return_patches:
        assert hq is not None, 'hq must be provided when return_patches=True'

    pred_weighted_sum = torch.zeros((C, H, W), device=device)
    weight_accumulator = torch.zeros((1, H, W), device=device)
    patch_list = []

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    lq_padded = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
    _, H_pad, W_pad = lq_padded.shape

    # Linear blending weight map: 1 at center, non-zero at edges
    center = (patch_size - 1) / 2
    floor = 0.05
    dist = (torch.arange(patch_size, device=device).float() - center).abs()
    ramp = 1 - dist / center
    ramp = ramp * (1 - floor) + floor
    weight_2d = ramp.view(-1, 1) * ramp.view(1, -1)

    model.eval()
    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = lq_padded[:, y:y+patch_size, x:x+patch_size]
                patch_input = patch.unsqueeze(0)
                patch_pred  = model(patch_input).squeeze(0)

                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                h_crop = y_end - y
                w_crop = x_end - x
                w = weight_2d[:h_crop, :w_crop]
                pred_weighted_sum[:, y:y_end, x:x_end] += patch_pred[:, :h_crop, :w_crop] * w
                weight_accumulator[:, y:y_end, x:x_end] += w

                if return_patches and y < H and x < W:
                    lq_patch = patch[:, :h_crop, :w_crop]
                    pred_patch = patch_pred[:, :h_crop, :w_crop]
                    hq_patch = hq[:, y:y_end, x:x_end]
                    if h_crop >= patch_size // 2 and w_crop >= patch_size // 2:
                        patch_list.append((lq_patch, hq_patch, pred_patch, y, x))

    weight_accumulator = weight_accumulator.clamp(min=1e-8)
    stitched = pred_weighted_sum / weight_accumulator

    if return_patches:
        return stitched, patch_list
    return stitched