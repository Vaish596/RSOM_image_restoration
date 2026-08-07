# Photoacoustic (RSOM) Image Restoration

Deep-learning restoration of photoacoustic (Raster-Scan Optoacoustic Microscopy, RSOM) images degraded by **undersampling artifacts**. The project trains supervised and generative models to map **low-quality (LQ)** reconstructions, produced by reconstructing from *undersampled raw acquisition data*, back to their **high-quality (HQ)** originals reconstructed from the *full* raw data.


---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Repository Structure](#repository-structure)
4. [The Task & Data](#the-task--data)
5. [Models](#models)
6. [Metrics & Visualisations](#metrics--visualisations)

---

## Overview

RSOM is a photoacoustic imaging modality. Full-dataset reconstruction is slow and memory-heavy, so acquisition often uses **undersampled scan positions**. This introduces characteristic streaking/ghosting artifacts. The goal here is to train neural networks that reconstruct the artifact-free image from the undersampled one, so acquisitions could theoretically be sped up.

Five model families are implemented (`model_type` in config):

| Model type | Family | Notes |
|---|---|---|
| `UNET` | CNN | Classic encoder–decoder U-Net with skip connections (supervised) |
| `ESRGAN` | GAN | RRDB generator + SN U-Net discriminator, perceptual (VGG) + adversarial loss |
| `HAT` | Transformer | Hybrid Attention Transformer (Swin windows + overlapping cross-attention), perceptual loss |
| `PALETTE` | Diffusion | Pixel-space conditional DDPM/DDIM restoration (PALETTE / SR3-style) |
| `DIP` | Zero-shot | Deep Image Prior — per-image optimisation, no checkpoint needed |


All training/evaluation is orchestrated with:
- **PyTorch Lightning 2.x** — `LightningModule` / `LightningDataModule` / `Trainer`
- **Hydra** — YAML config composition (`configs/`)
- **Weights & Biases** — experiment tracking, image logging, metrics

---

## Quick Start

### Environment
```bash
pip install -r requirements.txt          # includes torch 2.5.1+cu121, lightning 2.6.x, diffusers, lpips, ...
wandb login                               # optional
```

### Train
```bash
python train.py model=unet data=rsom          # UNet baseline
python train.py model=hat                     # transformer
python train.py model=esrgan trainer=gan      # GAN setup
python train.py model=palette                 # diffusion
```
Checkpoints under `checkpoints/<MODEL>/<run_name>_<wandb_id>/`. Hydra dot-notation overrides work, e.g. `model.loss_type=l1`, `data.folder_path=...`, `data.use_slices=true`.

### Evaluate
```bash
# edit configs/eval_config.yaml → add the checkpoint paths you want
python eval_pipeline.py
```
Outputs: `eval_results/<csv_dir>/*.csv`, PNG figures, W&B runs.

### 4. Plot results
```bash
python scripts/plot_model_comparison.py eval_results/<csv_dir>            # single ratio
python scripts/plot_model_comparison.py eval_results/MIP_25 eval_results/MIP_35 eval_results/MIP_50   # ratio comparison
python scripts/plot_radial_frequency.py eval_results/<csv_dir>/radial_profiles.csv
```
---

## Repository Structure

```
SupreRes/
├── train.py                  # Hydra entry point — builds run name, logger, callbacks, Trainer, fits + tests
├── pipeline.py               # SRLightningModel: unified LightningModule for every model_type
├── eval_pipeline.py          # checkpoint / LQ / DIP evaluation → CSVs, W&B, PNG visualisations + box plots
├── losses.py                 # SRLoss (l1/mse/l1_ssim), VGGFeatureLoss, gan_loss
├── requirements.txt
├── configs/                  # Hydra configs
│   ├── config.yaml           # defaults list + wandb settings
│   ├── model/{unet,esrgan,hat,palette,dip}.yaml
│   ├── data/rsom.yaml        # dataset + dataloader + slice/log-scale options
│   ├── trainer/{default,gan,hat}.yaml
│   ├── checkpoint/default.yaml
│   └── eval_config.yaml      # which checkpoints/analyses to evaluate
├── dataloader/
│   ├── datamodule.py         # RSOMDataModule (Lightning) picks dataset by flags
│   ├── rsom_dataset.py       # RSOMPatchDataset (MIP), RSOMSliceDataset (volumes), SuperResolutionDataset
│   ├── augmentations.py      # paired geometric (HQ+LQ) + LQ-only photometric augs
│   ├── patch_utils.py        # paired crops, informative-patch rejection, padding, sliding-window inference
│   ├── create_dataset.py     # raw-.mat → HQ/LQ pair dataset pipeline (undersample BEFORE reconstruction)
│   └── inference_utils.py    # plain overlap-averaged sliding window (older, uniform weights)
├── model/
│   ├── unet.py               # UNet2D (supervised)
│   ├── esrgan.py             # RRDBNet generator + UNetDiscriminatorSN
│   ├── hat.py                # HAT: Hybrid Attention Transformer (W-HTAB + OCA)
│   ├── palette.py            # PALETTE pixel-space conditional diffusion (DDPM/DDIM)
│   ├── dip.py                # Deep Image Prior (skip-net + per-image optimisation)
│   ├── ldm.py                # two-stage Latent Diffusion (VAE → denoising UNet) [experimental]
│   ├── indi.py               # InDI (iterative diffusion) [experimental]
│   └── helper/diffusion_unet.py  # time-conditioned U-Net shared by Palette/InDI
├── scripts/
│   ├── plot_model_comparison.py  # box/barch/table/pair/radar/corr/diverging/stat-from-CSVs
│   ├── plot_radial_frequency.py  # radial FFT comparison figure
├── Data/RSOM/                 # processed datasets (gitignored)
│   ├── processed_{25,35,50}/          # MIP datasets, 2D pairs
│   └── processed_slices_{25,35,50}/   # volume datasets, 4D (n_slices,H,W,3)
├── checkpoints/               # Lightning checkpoints per run (gitignored)
├── eval_results/              # CSV + PNG + radial profiles (gitignored)
```

---

## The Task & Data

### Data Structure

Each dataset root contains `train/`, `val/`, `test/` split folders. Every sample is a directory containing:

```
sample_name/
├── HQ.npy          # reconstruction from FULL raw data (ground truth)
├── LQ.npy          # reconstruction from UNDERSAMPLED raw data (input)
├── HQ_MIP.npy     # [slice datasets only] pre-computed GT maximum-intensity projection
├── mask.npy.        # the boolean undersampling mask
├── metadata.json   # sampling ratio, mask type, shapes, global percentiles, ...
└── comparison.png  # optional visualisation (HQ | LQ | difference)
```

**Arrays** are `float32`, normalised to `[0, 1]` via 1–99 % percentile clipping (percentiles computed from HQ and applied to LQ).

Two formats are used:

- **MIP datasets** (e.g. `processed_50`): samples are 2-D projections. `HQ.npy`/`LQ.npy` are `(H, W, 3)` RGB.
- **Slice datasets** (e.g. `processed_slices_50`): samples are full 3-D volumes. `HQ.npy`/`LQ.npy` are `(n_slices, H, W, 3)` typically `(168, 334, 771, 3)`; training samples single slices, validation/test process whole volumes and compute MIPs. Slice models are optionally trained in **log space** (`log1p(x * C)`).



## Models

### UNet2D (`model/unet.py`)
Encoder: 4 down-sampling levels, each `(Conv3→BN→ReLU)×2` with `MaxPool2d` between levels; the decoder upsamples + concatenates skips and halves the channel count through 3 up-blocks; a final 1×1 conv produces the output (no activation). Fully convolutional → any input size; uses `F.interpolate` for flexible upsampling.

### ESRGAN (`model/esrgan.py`)
- Generator: `RRDBNet` — 23 RRDB (Residual-in-Residual Dense Blocks) with 0.2 residual scaling.
- Discriminator: `UNetDiscriminatorSN` — U-Net style discriminator with spectral normalisation (Real-ESRGAN).
- Loss = pixel + VGG perceptual + adversarial (`gan_loss` in `losses.py`).

### HAT (`model/hat.py`)
Hybrid Attention Transformer:
- `RHAG` residual groups of alternating shifted/unshifted-window attention blocks (`HAB` / W-MSA) with channel attention (CAB) + an **Overlapping Cross-Attention (OCA)** block for cross-window information flow.
- Relative-position-biased attention; `PatchEmbed`/`PatchUnEmbed` flattening at patch_size 1.
- `upsampler=''` (identity) for the same-dimension restoration task (the `pixelshuffle` branch is for SR).

### PALETTE (`model/palette.py`)
Conditional diffusion in pixel space: input `concat(noisy_HQ, LQ)` (6 channels) → U-Net (from `model/helper/diffusion_unet.py`) predicts **velocity** (v-prediction). Training uses a `DDPMScheduler`; inference uses a faster `DDIMScheduler` (50 steps).

### DIP (`model/dip.py`)
Deep Image Prior: the network is **untrained** and optimised per image at inference. Classic encoder-decoder with skip connections (5 scales, LeakyReLU). `dip_optimise()` implements the paper’s default recipe: fixed noise input, per-step noise perturbation, exponential moving average of the output, backtracking on loss spikes, and optional multi-run averaging.


### Losses (`losses.py`)
| Loss | Definition |
|---|---|
| `SRLoss` | `'l1'`, `'mse'`, or `'l1_ssim'` (weighted L1 + `1−SSIM`) |
| `VGGFeatureLoss` | L2 on VGG19 `relu` features (perceptual) |
| `gan_loss` | LS-GAN MSE target (soft-label 0.9 real) |

Training/validation/test steps are implemented for every model family in `pipeline.py`, including slice-volume batch handling (`_predict_slices`) and log-space inversion before metric computation.


## Metrics & Visualisations

### 11 metrics per sample (`eval_results/` CSVs)

| Column | Meaning | Good |
|---|---|---|
| `psnr` | peak signal-to-noise | higher |
| `ssim` | structural similarity | higher |
| `lpips` | deep perceptual distance (Alex-Net) | lower |
| `epi` | Edge-Preservation Index (gradient correlation) | ≈1 |
| `vessel_dice` | vessel-mask Dice (Frangi) overlap | higher |
| `density_rel_error` | (pred−GT)/GT vessel pixel fraction | ≈0 |
| `comp_rel_error` | connected-component count deviation | ≈0 |
| `tortuosity_ratio` | pred/GT vessel winding (`skan`) | ≈1 |
| `err_vessel` | mean |pred−GT| inside vessels | lower |

`vessel_masks` via Frangi filter → percentile threshold → `remove_small_objects`.
