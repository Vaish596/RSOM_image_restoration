"""
rsom_dataset.py
===============
Loading strategy:
  ─ Training  : random patch extraction + augmentation per __getitem__ call.
  ─ Val / Test: full image returned as-is (no patch extraction, no augmentation).

repeat_factor:
  During training __len__ returns len(samples) × repeat_factor.  The extra
  iterations re-index into the same samples with fresh RNG seeds, generating
  different random crops and augmentations each time — effectively a free
  enlargement of the apparent dataset without storing extra data on disk.
"""

from __future__ import annotations
import os
import json
import torch
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from .augmentations import augment_pair
from .patch_utils import (
    maybe_pad_image,
    sample_informative_patch,
    validate_pair_shapes,
    validate_patch_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert a float32 numpy array to a PyTorch tensor in (C, H, W) layout.

    Channel layout convention:
      (H, W)    → (1, H, W)   grayscale, single channel
      (H, W, C) → (C, H, W)   RGB or multi-channel
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]        # H×W → 1×H×W
    else:
        arr = arr.transpose(2, 0, 1)       # H×W×C → C×H×W
    return torch.from_numpy(arr)


def _ensure_float32(arr: np.ndarray, name: str) -> np.ndarray:
    """Cast to float32 and warn if the source dtype differs."""
    if arr.dtype != np.float32:
        warnings.warn(
            f"{name} loaded with dtype {arr.dtype}; casting to float32.",
            stacklevel=3,
        )
        arr = arr.astype(np.float32)
    return arr


def _discover_samples(root: Path) -> List[Path]:
    """
    Walk root and collect every directory that contains both HQ.npy and LQ.npy.

    the scanner works whether samples are at root/sample/ or root/subdir/sample/.
    """
    found: List[Path] = []
    for candidate in sorted(root.rglob("HQ.npy")):
        sample_dir = candidate.parent
        if (sample_dir / "LQ.npy").exists():
            found.append(sample_dir)
        else:
            warnings.warn(f"Skipping {sample_dir}: LQ.npy not found.")
    return found


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RSOMPatchDataset(Dataset):
    """
    PyTorch Dataset for paired RSOM photoacoustic restoration.

    Args:
        root_dir:
            Path to one split directory, e.g. ``folder_path/train``.
            Every sub-directory containing ``HQ.npy`` and ``LQ.npy`` is
            treated as one sample.

        mode:
            ``"train"`` enables patch extraction and augmentation.
            ``"val"`` / ``"test"`` return the full image with no augmentation.

        patch_size:
            Side length (pixels) of the square training patches.
            Ignored in val/test mode.

        repeat_factor:
            Logical multiplier for ``__len__`` during training.
            E.g. repeat_factor=10 means each epoch draws 10 random patches
            per sample, while only one unique HQ/LQ pair is stored on disk.

        informative_threshold:
            Minimum mean pixel value for a patch to be accepted.
            Patches whose mean is below this threshold are re-sampled.

        max_patch_tries:
            Maximum re-sampling attempts before giving up and returning the
            last candidate (prevents infinite loops on mostly-empty images).

        augment_cfg:
            Dictionary of keyword arguments forwarded to ``augment_pair()``.
            Defaults to the recommended RSOM augmentation suite.
            Pass ``{}`` to disable all augmentations.

        load_metadata:
            If True, ``metadata.json`` is parsed and returned as an
            additional ``"metadata"`` key in the output dictionary.

        pad_if_needed:
            If True, images smaller than patch_size are zero-padded before
            crop extraction rather than raising an error.  Useful during
            early development when some volumes may be smaller than expected.

    Returns (per sample):
        dict with keys:
          ``"lq"``      : torch.Tensor (C, H, W) — input to the network
          ``"hq"``      : torch.Tensor (C, H, W) — restoration target
          ``"sample_id"``: str — name of the sample folder
          ``"metadata"`` : dict (only when load_metadata=True)
    """

    # Default augmentation config
    _DEFAULT_AUG_CFG: Dict = dict(
        use_hflip=True,
        use_vflip=True,
        use_rot90=True,
        use_transpose=True,
        use_noise=False,          # disabled by default; enable if helpful
        noise_std=0.01,
        use_intensity_jitter=False,
        brightness_range=0.05,
    )

    def __init__(
        self,
        root_dir: Union[str, Path],
        mode: str = "train",
        patch_size: int = 128,
        repeat_factor: int = 1,
        informative_threshold: float = 0.02,
        max_patch_tries: int = 20,
        augment_cfg: Optional[Dict] = None,
        load_metadata: bool = False,
        pad_if_needed: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir   = Path(root_dir)
        self.mode       = mode.lower()
        self.patch_size = patch_size
        self.repeat_factor      = max(1, repeat_factor)
        self.informative_threshold = informative_threshold
        self.max_patch_tries    = max_patch_tries
        self.load_metadata      = load_metadata
        self.pad_if_needed      = pad_if_needed
        self.is_train           = self.mode == "train"

        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be 'train', 'val', or 'test'; got '{mode}'.")

        # Resolve augmentation config
        self.augment_cfg: Dict = (
            self._DEFAULT_AUG_CFG.copy() if augment_cfg is None else augment_cfg
        )

        # Discover all valid sample directories
        self.samples: List[Path] = _discover_samples(self.root_dir)
        if not self.samples:
            raise FileNotFoundError(
                f"No valid sample directories found under '{self.root_dir}'. "
                "Each sample directory must contain HQ.npy and LQ.npy."
            )
        
        self._validate_first_sample()

        print(
            f"[RSOMPatchDataset] mode={self.mode} | "
            f"{len(self.samples)} samples | "
            f"patch_size={patch_size} | repeat_factor={self.repeat_factor}"
        )

    # ------------------------------------------------------------------
    # Length and indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Logical dataset length.

        During training, repeat_factor inflates __len__ so that more random
        patches are drawn per epoch.  Each call to __getitem__ generates a
        fresh random crop and augmentation, giving effectively more unique
        training samples per epoch.
        """
        base = len(self.samples)
        return base * self.repeat_factor if self.is_train else base

    def __getitem__(self, idx: int) -> Dict:
        # Map repeated indices back to the actual sample
        sample_idx = idx % len(self.samples)
        sample_dir = self.samples[sample_idx]

        rng = np.random.default_rng()

        # --- Load arrays ---
        hq, lq = self._load_pair(sample_dir)

        # --- Patch extraction (training only) ---
        if self.is_train:
            if self.pad_if_needed:
                hq = maybe_pad_image(hq, self.patch_size)
                lq = maybe_pad_image(lq, self.patch_size)
            else:
                validate_patch_size(self.patch_size, hq, sample_id=sample_dir.name)

            hq, lq = sample_informative_patch(
                hq, lq,
                patch_size=self.patch_size,
                threshold=self.informative_threshold,
                max_tries=self.max_patch_tries,
                rng=rng,
            )

            # --- Augmentation (training only) ---
            hq, lq = augment_pair(hq, lq, rng=rng, **self.augment_cfg)

        # --- Convert to tensors ---
        y = _to_tensor(hq)
        x = _to_tensor(lq)

        # output: Dict = {
        #     "lq":        x,     # network input
        #     "hq":        y,     # restoration target
        #     "sample_id": sample_dir.name,
        # }

        # # --- Optional metadata ---
        # if self.load_metadata:
        #     output["metadata"] = self._load_metadata(sample_dir)

        return x,y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pair(self, sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load, validate and cast HQ/LQ arrays."""
        hq_path = sample_dir / "HQ.npy"
        lq_path = sample_dir / "LQ.npy"

        # Existence check (redundant after discovery, but makes errors clear)
        if not hq_path.exists():
            raise FileNotFoundError(f"HQ.npy missing: {hq_path}")
        if not lq_path.exists():
            raise FileNotFoundError(f"LQ.npy missing: {lq_path}")

        hq = np.load(hq_path)
        lq = np.load(lq_path)

        hq = _ensure_float32(hq, f"{sample_dir.name}/HQ.npy")
        lq = _ensure_float32(lq, f"{sample_dir.name}/LQ.npy")

        # Shape sanity: both must match exactly
        validate_pair_shapes(hq, lq, sample_id=sample_dir.name)

        # Dimension sanity: must be 2-D or 3-D
        if hq.ndim not in {2, 3}:
            raise ValueError(
                f"[{sample_dir.name}] Expected 2-D or 3-D array; got shape {hq.shape}."
            )

        return hq, lq

    def _load_metadata(self, sample_dir: Path) -> Dict:
        """Parse metadata.json; return empty dict if absent."""
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            warnings.warn(f"metadata.json not found in {sample_dir}; returning {{}}.")
            return {}
        with open(meta_path, "r") as f:
            return json.load(f)

    def _validate_first_sample(self) -> None:
        """
        Eagerly load the first sample to surface shape / file errors
        before training begins.  Raises on any detected issue.
        """
        sample_dir = self.samples[0]
        hq, lq = self._load_pair(sample_dir)

        if self.is_train:
            min_dim = min(hq.shape[0], hq.shape[1])
            if self.patch_size > min_dim and not self.pad_if_needed:
                raise ValueError(
                    f"patch_size={self.patch_size} > smallest image dimension "
                    f"({min_dim}) in first sample '{sample_dir.name}'. "
                    "Enable pad_if_needed or reduce patch_size."
                )

        print(
            f"[RSOMPatchDataset] First sample '{sample_dir.name}' — "
            f"HQ: {hq.shape} {hq.dtype}, range [{hq.min():.3f}, {hq.max():.3f}]"
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_sample_ids(self) -> List[str]:
        """Return the list of sample folder names (useful for bookkeeping)."""
        return [s.name for s in self.samples]
    


class SuperResolutionDataset(Dataset):
    def __init__(
            self,
        folder_path:  str,
        scale_factor: int        = 4,
        img_channels: int        = 3,
        hr_size:      int | None = None
    ):
        """
        folder_path: folder containing HR images
        scale_factor: factor to downscale for LR images
        """
        self.image_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in: {folder_path}")

        self.scale_factor = scale_factor
        self.img_channels = img_channels
        self.hr_size      = hr_size

        assert img_channels in (1, 3), "img_channels must be 1 (grayscale) or 3 (RGB)"

    def __len__(self):
        return len(self.image_files)
    
    def _load_image(self, path: str) -> Image.Image:
        """Load and convert image to the correct colour mode."""
        img = Image.open(path)
        if self.img_channels == 1:
            return img.convert("L")
        else:
            return img.convert("RGB")
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL Image → float32 tensor in [0, 1].
        Grayscale: (1, H, W)   RGB: (3, H, W)
        """
        arr = np.array(img, dtype=np.float32) / 255.0
        t   = torch.from_numpy(arr)
        if self.img_channels == 1:
            return t.unsqueeze(0)          # (1, H, W)
        else:
            return t.permute(2, 0, 1)      # (3, H, W)

    def __getitem__(self, idx):
        img_hr = self._load_image(self.image_files[idx])
        # 1. Resize to canonical HR size
        if self.hr_size is not None:
            img_hr = img_hr.resize(
                (self.hr_size, self.hr_size),   # PIL uses (W, H)
                Image.BICUBIC,
            )

        # 3. Derive LR by downscaling HR patch
        lr_w = img_hr.width  // self.scale_factor
        lr_h = img_hr.height // self.scale_factor
        img_lr = img_hr.resize((lr_w, lr_h), Image.BICUBIC)
        # img_lr = img_lr.resize((img_hr.width, img_hr.height), Image.BICUBIC) #bicubic upsampling to get same dims as hr

        # convert to tensor.unsqueeze(0)
        x = self._to_tensor(img_lr)
        y = self._to_tensor(img_hr)
        

        return x, y


class RSOMSliceDataset(Dataset):
    """
    PyTorch Dataset for paired RSOM photoacoustic restoration using full
    volume slices (Y, H, W, 3) instead of a single MIP image.

    Training:
        Loads the full volume (Y, H, W, 3) once, caches it, then each
        __getitem__ picks a random slice, extracts a patch, augments,
        and returns (lq, hq) — same shape as RSOMPatchDataset so the
        training_step needs zero changes.

    Validation / Test:
        Returns the full volume as stacked tensors plus the pre-computed
        GT MIP, so the pipeline can compute both per-slice and MIP metrics.

    Log scaling:
        When log_scale_factor > 0, applies log1p(x * C) to the loaded
        arrays BEFORE patch extraction / augmentation.  The model thus
        trains in log space.  Validation inverts before metric computation.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        mode: str = "train",
        patch_size: int = 128,
        repeat_factor: int = 1,
        informative_threshold: float = 0.02,
        max_patch_tries: int = 20,
        augment_cfg: Optional[Dict] = None,
        load_metadata: bool = False,
        pad_if_needed: bool = True,
        # ---- Slice-specific params ---------------------------------------- #
        n_slices: int = 168,
        use_log_scale: bool = False,
        log_scale_factor: float = 0.0,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.mode = mode.lower()
        self.patch_size = patch_size
        self.repeat_factor = max(1, repeat_factor)
        self.informative_threshold = informative_threshold
        self.max_patch_tries = max_patch_tries
        self.load_metadata = load_metadata
        self.pad_if_needed = pad_if_needed
        self.is_train = self.mode == "train"
        self.n_slices = n_slices
        self.use_log_scale = use_log_scale
        self.log_scale_C = log_scale_factor

        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be 'train', 'val', or 'test'; got '{mode}'.")

        self.augment_cfg: Dict = (
            self._DEFAULT_AUG_CFG.copy() if augment_cfg is None else augment_cfg
        )

        self.samples: List[Path] = _discover_samples(self.root_dir)
        if not self.samples:
            raise FileNotFoundError(
                f"No valid sample directories found under '{self.root_dir}'. "
                "Each sample directory must contain HQ.npy and LQ.npy."
            )

        self._volume_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._validate_first_sample()

        print(
            f"[RSOMSliceDataset] mode={self.mode} | "
            f"{len(self.samples)} samples | "
            f"n_slices={n_slices} | repeat_factor={self.repeat_factor} | "
            f"log_scale={'log1p(C*x)' if self.use_log_scale and self.log_scale_C else 'none'}"
        )

    # Default augmentation config
    _DEFAULT_AUG_CFG: Dict = dict(
        use_hflip=True,
        use_vflip=True,
        use_rot90=True,
        use_transpose=True,
        use_noise=False,
        noise_std=0.01,
        use_intensity_jitter=False,
        brightness_range=0.05,
    )


    def __len__(self) -> int:
        base = len(self.samples)
        if self.is_train:
            return base * self.n_slices * self.repeat_factor
        return base

    def _resolve_idx(self, idx: int) -> Tuple[Path, int]:
        """Map flat training index → (sample_dir, slice_idx)."""
        per_sample = self.n_slices * self.repeat_factor
        sample_idx = idx // per_sample
        remainder = idx % per_sample
        slice_idx = remainder // self.repeat_factor
        return self.samples[sample_idx], slice_idx


    def _load_volume(self, sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        key = sample_dir.name
        if key not in self._volume_cache:
            hq = np.load(sample_dir / "HQ.npy")  # (Y, H, W, 3)
            lq = np.load(sample_dir / "LQ.npy")
            hq = _ensure_float32(hq, f"{key}/HQ.npy")
            lq = _ensure_float32(lq, f"{key}/LQ.npy")
            validate_pair_shapes(hq, lq, sample_id=key)
            if hq.ndim != 4:
                raise ValueError(f"[{key}] Expected 4-D array (Y,H,W,C); got shape {hq.shape}.")
            self._volume_cache[key] = (hq, lq)
        return self._volume_cache[key]


    def __getitem__(self, idx: int):
        if self.is_train:
            sample_dir, slice_idx = self._resolve_idx(idx)
            rng = np.random.default_rng()
        else:
            sample_dir = self.samples[idx % len(self.samples)]
            rng = None  # not used for val/test

        # Load volume
        hq_vol, lq_vol = self._load_volume(sample_dir)  # (Y, H, W, 3)

        # ---- Log scaling (applied to numpy before any crop/augment) ----
        if self.log_scale_C and self.use_log_scale:
            hq_vol_log = np.log1p(hq_vol * self.log_scale_C)
            lq_vol_log = np.log1p(lq_vol * self.log_scale_C)
        else:
            hq_vol_log = hq_vol
            lq_vol_log = lq_vol

        if self.is_train:
            # Pick a slice
            hq = hq_vol_log[slice_idx]  # (H, W, 3)
            lq = lq_vol_log[slice_idx]

            # Pad if needed
            if self.pad_if_needed:
                hq = maybe_pad_image(hq, self.patch_size)
                lq = maybe_pad_image(lq, self.patch_size)
            else:
                validate_patch_size(self.patch_size, hq, sample_id=sample_dir.name)

            # Paired random crop
            hq, lq = sample_informative_patch(
                hq, lq,
                patch_size=self.patch_size,
                threshold=self.informative_threshold,
                max_tries=self.max_patch_tries,
                rng=rng,
            )

            # Augment (in log space — geometric only, identical for HQ/LQ)
            hq, lq = augment_pair(hq, lq, rng=rng, **self.augment_cfg)

            # Convert to tensors
            y = _to_tensor(hq)  # (3, 128, 128)
            x = _to_tensor(lq)
            return x, y

        else:
            # Validation / test: return full volume + pre-computed GT MIP
            Y = hq_vol_log.shape[0]
            lq_tensor = torch.stack([_to_tensor(lq_vol_log[s]) for s in range(Y)])  # (Y, 3, H, W)
            hq_tensor = torch.stack([_to_tensor(hq_vol_log[s]) for s in range(Y)])  # (Y, 3, H, W)
            # GT MIP from pre-computed HQ_MIP.npy (independently normalized)
            hq_mip = _to_tensor(np.load(sample_dir / "HQ_MIP.npy"))  # (3, H, W)

            return lq_tensor, hq_tensor, hq_mip

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_first_sample(self) -> None:
        sample_dir = self.samples[0]
        hq_vol, _ = self._load_volume(sample_dir)
        if self.is_train:
            min_dim = min(hq_vol.shape[1], hq_vol.shape[2])
            if self.patch_size > min_dim and not self.pad_if_needed:
                raise ValueError(
                    f"patch_size={self.patch_size} > smallest spatial dimension "
                    f"({min_dim}) in first sample '{sample_dir.name}'. "
                    "Enable pad_if_needed or reduce patch_size."
                )
        print(
            f"[RSOMSliceDataset] First sample '{sample_dir.name}' — "
            f"volume: {hq_vol.shape} {hq_vol.dtype}, "
            f"range [{hq_vol.min():.3f}, {hq_vol.max():.3f}]"
        )

    def get_sample_ids(self) -> List[str]:
        return [s.name for s in self.samples]
