"""
datamodule.py
=============
PyTorch Lightning DataModule for RSOM photoacoustic restoration.

Responsibility:
  Owns the construction of all three Dataset objects (train / val / test)
  and their corresponding DataLoaders.  Accepts a single configuration
  object so that the entire data pipeline can be reproduced from a config
  file (e.g. Hydra / YAML).

Split layout on disk::

    folder_path/
    ├── train/   ← used by self.train_dataset
    ├── val/     ← used by self.val_dataset
    └── test/    ← used by self.test_dataset

Usage with PyTorch Lightning Trainer::

    dm = RSOMDataModule(folder_path="data/", batch_size=8, patch_size=128)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, datamodule=dm)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from .rsom_dataset import RSOMPatchDataset, SuperResolutionDataset, RSOMSliceDataset


class RSOMDataModule(L.LightningDataModule):
    """
    Lightning DataModule wrapping RSOMPatchDataset for all three splits.

    Args:
        folder_path:
            Root directory containing ``train/``, ``val/``, and ``test/``
            subdirectories.

        batch_size:
            Number of samples per training batch.

        patch_size:
            Spatial size (pixels) of random training patches.  Val and test
            always use the full image.

        num_workers:
            Number of parallel worker processes for DataLoader.
            Set to 0 for in-process loading (easier debugging).

        pin_memory:
            Pin DataLoader memory for faster GPU transfer.  Set to True when
            training on GPU.

        repeat_factor:
            Training epoch length multiplier.  ``len(train_set) * repeat_factor``
            random patches are drawn per epoch without additional disk reads.

        augment_cfg:
            Keyword arguments forwarded to ``augment_pair()`` via the Dataset.
            ``None`` uses the recommended RSOM defaults.

        informative_threshold:
            Minimum mean intensity of a training patch for it to be accepted.

        max_patch_tries:
            Maximum random crop attempts before accepting an uninformative patch.

        load_metadata:
            Whether to include the parsed ``metadata.json`` in each batch.

        pad_if_needed:
            Pad images smaller than ``patch_size`` instead of raising an error.

        val_batch_size:
            Batch size for validation and test.  Defaults to ``batch_size``.
            Full-image batches are often larger than GPU VRAM allows at training
            batch size — set this to 1 for safety.
    """

    def __init__(
        self,
        folder_path:            Union[str, Path],
        batch_size:              int   = 8,
        patch_size:              int   = 128,
        pin_memory:              bool  = True,
        num_workers:             int   = 2,
        repeat_factor:           int   = 10,
        augment_cfg:             Optional[Dict[str, Any]] = None,
        informative_threshold:   float = 0.02,
        max_patch_tries:         int   = 20,
        load_metadata:           bool  = False,
        pad_if_needed:           bool  = True,
        val_batch_size:          Optional[int] = None,
        pre_split:               bool  = True,
        # -------- For other datasets ----------
        img_channels:            int        = 3,
        hr_size:                 int | None = 768,
        scale_factor:            int        = 4,
        val_split:               float      = 0.2,
        test_split:              float      = 0.1,
        # -------- Slice dataset params ----------
        use_slices:              bool  = False,
        n_slices:                int   = 168,
        use_log_scale:           bool  = False,
        log_scale_factor:        float = 0.0,
        
    ) -> None:
        super().__init__()

        self.folder_path          = Path(folder_path)
        self.batch_size            = batch_size
        self.patch_size            = patch_size
        self.num_workers           = num_workers
        self.pin_memory            = pin_memory
        self.repeat_factor         = repeat_factor
        self.augment_cfg           = augment_cfg
        self.informative_threshold = informative_threshold
        self.max_patch_tries       = max_patch_tries
        self.load_metadata         = load_metadata
        self.pad_if_needed         = pad_if_needed
        self.val_batch_size        = val_batch_size if val_batch_size is not None else batch_size
        self._pre_split            = pre_split
        # old dataset
        self.scale_factor          = scale_factor
        self.val_split             = val_split
        self.test_split            = test_split
        self.img_channels          = img_channels
        self.hr_size               = hr_size
        # slice dataset
        self.use_slices            = use_slices
        self.n_slices              = n_slices
        self.use_log_scale         = use_log_scale
        self.log_scale_factor      = log_scale_factor
        # These are set in setup()
        self.train_dataset: Optional[Union[RSOMPatchDataset, Dataset]] = None
        self.val_dataset:   Optional[Union[RSOMPatchDataset, Dataset]] = None
        self.test_dataset:  Optional[Union[RSOMPatchDataset, Dataset]] = None

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self.use_slices:
            self._setup_slices(stage)
        elif self._pre_split:
            if stage in {"fit", None}:
                self.train_dataset = RSOMPatchDataset(
                    root_dir              = self.folder_path / "train",
                    mode                  = "train",
                    patch_size            = self.patch_size,
                    repeat_factor         = self.repeat_factor,
                    informative_threshold = self.informative_threshold,
                    max_patch_tries       = self.max_patch_tries,
                    augment_cfg           = self.augment_cfg,
                    load_metadata         = self.load_metadata,
                    pad_if_needed         = self.pad_if_needed,
                )

                self.val_dataset = RSOMPatchDataset(
                    root_dir      = self.folder_path / "val",
                    mode          = "val",
                    load_metadata = self.load_metadata,
                )

            if stage in {"validate", None} and self.val_dataset is None:
                self.val_dataset = RSOMPatchDataset(
                    root_dir      = self.folder_path / "val",
                    mode          = "val",
                    load_metadata = self.load_metadata,
                )

            if stage in {"test", None}:
                self.test_dataset = RSOMPatchDataset(
                    root_dir      = self.folder_path / "test",
                    mode          = "test",
                    load_metadata = self.load_metadata,
                )
        else:
            # ---------- Layout A: flat folder, random split ----------
            full_dataset = SuperResolutionDataset(
                folder_path=self.folder_path,
                scale_factor=self.scale_factor,
                img_channels=self.img_channels,
                hr_size=self.hr_size,
            )
            n       = len(full_dataset)
            n_val   = max(1,round(self.val_split  * n))
            n_test  = max(1,round(self.test_split * n))
            n_train = n - n_val - n_test

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(42),
            )

    def _setup_slices(self, stage: Optional[str] = None) -> None:
        """Create RSOMSliceDataset instances."""
        ds_kwargs = dict(
            patch_size=self.patch_size,
            n_slices=self.n_slices,
            use_log_scale=self.use_log_scale,
            log_scale_factor=self.log_scale_factor,
            informative_threshold=self.informative_threshold,
            max_patch_tries=self.max_patch_tries,
            augment_cfg=self.augment_cfg,
            load_metadata=self.load_metadata,
            pad_if_needed=self.pad_if_needed,
        )
        if stage in {"fit", None}:
            self.train_dataset = RSOMSliceDataset(
                root_dir=self.folder_path / "train", mode="train",
                repeat_factor=self.repeat_factor, **ds_kwargs,
            )
            self.val_dataset = RSOMSliceDataset(
                root_dir=self.folder_path / "val", mode="val", **ds_kwargs,
            )
        if stage in {"validate", None} and self.val_dataset is None:
            self.val_dataset = RSOMSliceDataset(
                root_dir=self.folder_path / "val", mode="val", **ds_kwargs,
            )
        if stage in {"test", None}:
            self.test_dataset = RSOMSliceDataset(
                root_dir=self.folder_path / "test", mode="test", **ds_kwargs,
            )


    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = True,
            persistent_workers = (self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size  = self.val_batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = False,
            persistent_workers = (self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size  = self.val_batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = False,
            persistent_workers = (self.num_workers > 0),
        )

    # # ------------------------------------------------------------------
    # # Convenience
    # # ------------------------------------------------------------------

    # def summary(self) -> None:
    #     """Print a human-readable summary of the configured data pipeline."""
    #     print("=" * 60)
    #     print("RSOMDataModule Summary")
    #     print("=" * 60)
    #     print(f"  folder_path       : {self.folder_path}")
    #     print(f"  patch_size         : {self.patch_size}")
    #     print(f"  batch_size (train) : {self.batch_size}")
    #     print(f"  batch_size (val)   : {self.val_batch_size}")
    #     print(f"  num_workers        : {self.num_workers}")
    #     print(f"  repeat_factor      : {self.repeat_factor}")
    #     print(f"  informative thresh : {self.informative_threshold}")

    #     if self.train_dataset:
    #         print(f"  train samples      : {len(self.train_dataset.samples)}")
    #         print(f"  train __len__      : {len(self.train_dataset)}")
    #     if self.val_dataset:
    #         print(f"  val samples        : {len(self.val_dataset.samples)}")
    #     if self.test_dataset:
    #         print(f"  test samples       : {len(self.test_dataset.samples)}")
    #     print("=" * 60)