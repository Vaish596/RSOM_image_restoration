import os
import signal
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from pipeline import SRLightningModel
from dataloader.datamodule import RSOMDataModule


# ============================================================
#  Checkpoint callbacks
# ============================================================

def build_checkpoint_callbacks(model_type: str, ckpt_dir_name: str, save_every_n_epochs: int = 5):
    ckpt_dir = os.path.join("checkpoints", model_type, ckpt_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "periodic"), exist_ok=True)

    best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
        verbose=True,
    )

    last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        save_weights_only=False,
        verbose=False,
    )

    periodic = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "periodic"),
        filename="epoch={epoch:04d}",
        every_n_epochs=save_every_n_epochs,
        save_top_k=-1,
        save_weights_only=False,
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [best, last, periodic, lr_monitor], ckpt_dir


# ============================================================
#  Emergency save — unchanged
# ============================================================

class EmergencySaveCallback(L.Callback):
    def __init__(self, ckpt_dir: str):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self._trainer = None
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def setup(self, trainer, pl_module, stage):
        self._trainer = trainer

    def _handle_sigterm(self, signum, frame):
        print("\n[EmergencySave] SIGTERM received — saving emergency checkpoint...")
        self._save(self._trainer)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print("\n[EmergencySave] KeyboardInterrupt — saving emergency checkpoint...")
            self._save(trainer)

    def _save(self, trainer):
        if trainer is None:
            return
        path = os.path.join(self.ckpt_dir, "emergency.ckpt")
        trainer.save_checkpoint(path)
        print(f"[EmergencySave] Saved → {path}")


# ============================================================
#  Resume path resolution — unchanged logic, new signature
# ============================================================

def resolve_resume_path(
    resume_from_checkpoint: str | None,
    auto_resume: bool,
    ckpt_dir: str
) -> str | None:
    if resume_from_checkpoint is not None:
        if not os.path.isfile(resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from_checkpoint}")
        print(f"[train.py] Resuming from: {resume_from_checkpoint}")
        return resume_from_checkpoint

    if auto_resume:
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.isfile(last):
            print(f"[train.py] Auto-resuming from: {last}")
            return last
        print("[train.py] --auto_resume set but no last.ckpt found — starting fresh")

    return None


# ============================================================
#  Main — now driven by Hydra
# ============================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:  
    model_type = cfg.model.model_type

    # ---- Build a human-readable run name from key hyperparams ------------ #
    # This is what appears in W&B UI and checkpoint folder names
    run_name = (
        f"{model_type}"
        f"_lr{cfg.model.lr}"
        f"_bs{cfg.data.batch_size}"
        f"_loss-{cfg.model.loss_type}"
        f"_ep{cfg.trainer.max_epochs}"
        f"_data{cfg.data.get('folder_path', '').split('/')[-1]}"  # dataset name
    )
    # ---- W&B Logger ------------------------------------------------------- #
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        group=cfg.wandb.get("group", model_type),   # group by architecture
        tags=[
            model_type,
            cfg.model.loss_type,
            cfg.data.get("folder_path", "").split("/")[-1],  # dataset name
            *cfg.wandb.get("tags", []),             # any extra tags from config
        ],
        notes=cfg.wandb.get("notes", ""),
        config=OmegaConf.to_container(cfg, resolve=True),  # full config stored in W&B
        log_model=False,   # set True to upload checkpoints to W&B as artifacts
    )

    wandb_run_id = logger.experiment.id

    # ---- Callbacks + checkpoint dir --------------------------------------- #
    save_every = cfg.checkpoint.save_every_n_epochs
    ckpt_dir_name = f"{run_name}_{wandb_run_id}"
    callbacks, ckpt_dir = build_checkpoint_callbacks(model_type, ckpt_dir_name, save_every)
    callbacks.append(EmergencySaveCallback(ckpt_dir))

    # ---- Resume ----------------------------------------------------------- #
    resume_path = resolve_resume_path(
        cfg.get("resume_from_checkpoint", None),
        cfg.get("auto_resume", False),
        ckpt_dir,
    )

    # ---- Model + Data ----------------------------------------------------- #
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg['use_slices'] = cfg.data.get('use_slices', False)
    model_cfg['use_log_scale'] = cfg.data.get('use_log_scale', False)
    model_cfg['log_scale_factor'] = cfg.data.get('log_scale_factor', 0.0)
    model = SRLightningModel(**model_cfg)
    # model      = SRLightningModel(**OmegaConf.to_container(cfg.model, resolve=True))
    datamodule = RSOMDataModule(**OmegaConf.to_container(cfg.data, resolve=True))

    # ---- Trainer ---------------------------------------------------------- #
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )

    # ---- Fit -------------------------------------------------------------- #
    # if not cfg.get("test_only", False):
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)

    final_path = os.path.join(ckpt_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"[train.py] Training complete. Final checkpoint → {final_path}")

    # ---- Test ------------------------------------------------------------- #
    test_ckpt = cfg.get("test_checkpoint", None)
    if test_ckpt is None:
        test_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    ckpt_for_test = test_ckpt if os.path.isfile(test_ckpt) else None
    trainer.test(model, datamodule, ckpt_path=ckpt_for_test)


if __name__ == "__main__":
    train()