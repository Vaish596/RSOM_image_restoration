"""
augmentations.py
================
Paired geometric and photometric augmentations for RSOM restoration training.

Design contract (CRITICAL):
  Every function that accepts (hq, lq) applies the IDENTICAL transformation
  to both arrays.  Breaking this contract would destroy the spatial correspondence
  required for supervised inverse-problem learning and silently corrupt training.

Augmentation philosophy for photoacoustic restoration:
  - Geometric augmentations are safe and encouraged: RSOM images have no
    preferred orientation, so flips and 90° rotations are always valid.
  - Intensity jitter on LQ simulates additional noise from the acquisition
    pipeline; it is applied only to LQ, never to HQ (the ground truth).
  - Heavy photometric distortion (colour jitter, large brightness shifts) is
    NOT applied because normalised photoacoustic intensities carry physical
    meaning.

All functions operate on float32 numpy arrays shaped (H, W) or (H, W, C)
and return the same dtype and layout.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Type alias for a paired image tuple
# ---------------------------------------------------------------------------
Pair = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Geometric augmentations (applied identically to HQ and LQ)
# ---------------------------------------------------------------------------

def random_hflip_pair(hq: np.ndarray, lq: np.ndarray, p: float = 0.5,
                      rng: Optional[np.random.Generator] = None) -> Pair:
    """
    Randomly flip both images horizontally (left ↔ right).

    The flip axis is the last spatial axis (axis=1 for H×W or H×W×C),
    which corresponds to the lateral scan direction in RSOM.

    Args:
        hq, lq: Paired float32 arrays (H,W) or (H,W,C).
        p:      Probability of applying the flip.
        rng:    Optional numpy Generator.

    Returns:
        (hq, lq) — flipped or unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        hq = np.flip(hq, axis=1).copy()
        lq = np.flip(lq, axis=1).copy()
    return hq, lq


def random_vflip_pair(hq: np.ndarray, lq: np.ndarray, p: float = 0.5,
                      rng: Optional[np.random.Generator] = None) -> Pair:
    """
    Randomly flip both images vertically (top ↔ bottom).

    The flip axis is axis=0 (the depth / axial axis in RSOM cross-sections).

    Args:
        hq, lq: Paired float32 arrays.
        p:      Flip probability.
        rng:    Optional numpy Generator.

    Returns:
        (hq, lq)
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        hq = np.flip(hq, axis=0).copy()
        lq = np.flip(lq, axis=0).copy()
    return hq, lq


def random_rot90_pair(hq: np.ndarray, lq: np.ndarray, p: float = 0.5,
                      rng: Optional[np.random.Generator] = None) -> Pair:
    """
    Randomly rotate both images by a multiple of 90 degrees.

    Rotation is applied in the H–W plane (axes 0 and 1), preserving the
    channel axis.  The rotation count k is sampled uniformly from {1, 2, 3}
    (0 would be a no-op, handled by the probability gate).

    Args:
        hq, lq: Paired float32 arrays (H,W) or (H,W,C).
        p:      Probability of applying any rotation.
        rng:    Optional numpy Generator.

    Returns:
        (hq, lq) — rotated or unchanged.

    Note:
        After rotation, H and W may be swapped.  This is intentional and
        consistent: patch-based training handles variable-sized inputs,
        and full-image evaluation typically sees the original orientation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        k = int(rng.integers(1, 4))           # 1, 2, or 3 quarter-turns
        hq = np.rot90(hq, k=k, axes=(0, 1)).copy()
        lq = np.rot90(lq, k=k, axes=(0, 1)).copy()
    return hq, lq


def random_transpose_pair(hq: np.ndarray, lq: np.ndarray, p: float = 0.5,
                          rng: Optional[np.random.Generator] = None) -> Pair:
    """
    Randomly transpose the spatial dimensions of both images.

    Transposing swaps H and W (axis 0 ↔ axis 1).  Combined with flips this
    generates all 8 elements of the dihedral group D4, doubling the effective
    training set diversity for square patches.

    Args:
        hq, lq: Paired float32 arrays.
        p:      Transpose probability.
        rng:    Optional numpy Generator.

    Returns:
        (hq, lq)
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        if hq.ndim == 2:
            hq = hq.T.copy()
            lq = lq.T.copy()
        else:
            # (H, W, C) → (W, H, C)
            hq = np.transpose(hq, (1, 0, 2)).copy()
            lq = np.transpose(lq, (1, 0, 2)).copy()
    return hq, lq


# ---------------------------------------------------------------------------
# Photometric augmentations (LQ-only or paired, clearly labelled)
# ---------------------------------------------------------------------------

def add_noise_lq(lq: np.ndarray, noise_std: float = 0.01,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add small Gaussian noise to the LQ image only.

    This simulates additional measurement noise not captured by the
    undersampling mask and encourages the restoration model to be robust
    to slight input perturbations.  Applied to LQ only — HQ (ground truth)
    must not be corrupted.

    Args:
        lq:        LQ float32 array, values in [0, 1].
        noise_std: Standard deviation of the Gaussian noise.
        rng:       Optional numpy Generator.

    Returns:
        Noisy LQ, clipped to [0, 1] and same dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=noise_std, size=lq.shape).astype(lq.dtype)
    return np.clip(lq + noise, 0.0, 1.0)


def random_intensity_jitter_lq(lq: np.ndarray,
                                brightness_range: float = 0.05,
                                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply a small random global brightness offset to the LQ image only.

    Mimics slight gain drift in the acquisition electronics.  The shift is
    small enough to stay within [0, 1] for typical photoacoustic images.

    Args:
        lq:               LQ float32 array.
        brightness_range: Maximum absolute shift (±brightness_range).
        rng:              Optional numpy Generator.

    Returns:
        Shifted and clipped LQ array.
    """
    if rng is None:
        rng = np.random.default_rng()
    shift = float(rng.uniform(-brightness_range, brightness_range))
    return np.clip(lq + shift, 0.0, 1.0).astype(lq.dtype)


# ---------------------------------------------------------------------------
# Master augmentation function — call this from the dataset __getitem__
# ---------------------------------------------------------------------------

def augment_pair(
    hq: np.ndarray,
    lq: np.ndarray,
    # Geometric augmentation flags
    use_hflip:        bool = True,
    use_vflip:        bool = True,
    use_rot90:        bool = True,
    use_transpose:    bool = True,
    # Photometric augmentation flags (LQ-only)
    use_noise:        bool = False,
    noise_std:        float = 0.01,
    use_intensity_jitter: bool = False,
    brightness_range: float = 0.05,
    # Reproducibility
    rng: Optional[np.random.Generator] = None,
) -> Pair:
    """
    Apply the full suite of paired augmentations to an HQ/LQ image pair.

    Geometric transforms are applied identically to both arrays.
    Photometric transforms (noise, jitter) are applied ONLY to LQ.

    Args:
        hq, lq:            Paired float32 arrays (H,W) or (H,W,C).
        use_hflip:         Enable random horizontal flip.
        use_vflip:         Enable random vertical flip.
        use_rot90:         Enable random 90° rotation.
        use_transpose:     Enable random H/W transpose.
        use_noise:         Enable Gaussian noise on LQ.
        noise_std:         Std-dev of the LQ noise.
        use_intensity_jitter: Enable brightness jitter on LQ.
        brightness_range:  Max absolute brightness shift for LQ.
        rng:               Shared numpy Generator for all operations.
                           Providing the same rng across calls with a fixed
                           seed gives reproducible augmentations.

    Returns:
        (augmented_hq, augmented_lq)
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Geometric (HQ and LQ together) ---
    if use_hflip:
        hq, lq = random_hflip_pair(hq, lq, rng=rng)
    if use_vflip:
        hq, lq = random_vflip_pair(hq, lq, rng=rng)
    if use_rot90:
        hq, lq = random_rot90_pair(hq, lq, rng=rng)
    if use_transpose:
        hq, lq = random_transpose_pair(hq, lq, rng=rng)

    # --- Photometric (LQ only — HQ must remain unmodified) ---
    if use_noise:
        lq = add_noise_lq(lq, noise_std=noise_std, rng=rng)
    if use_intensity_jitter:
        lq = random_intensity_jitter_lq(lq, brightness_range=brightness_range, rng=rng)

    return hq, lq
