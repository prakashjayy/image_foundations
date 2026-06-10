"""
DINO training loop — PyTorch Lightning 2.x

Schedules (paper Sec 3 / Appendix D):
  LR          : linear warmup 10 ep  → cosine decay to min_lr
  Weight decay: cosine schedule 0.04 → 0.4  (per epoch)
  Teacher τ_t : linear warmup 0.04  → 0.07  (per epoch)
  Teacher EMA λ: cosine schedule 0.996 → 1.0 (per STEP — more faithful to paper)

Validation (every epoch):
  - DINO cross-entropy loss on the validation split
  - Every knn_eval_every_n_epochs: weighted k-NN accuracy on clean features
    (paper Appendix F.1 — no finetuning, just cosine-sim voting)

Checkpointing: top-3 by val/knn_accuracy + last.ckpt
"""

import math
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dino.config import DINOConfig, cfg as default_cfg
from dino.ds import MNISTMultiCrop, collate_multicrop
from dino.loss import DINOLoss
from dino.network import build_student_teacher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_schedule(step: int, total: int, start: float, end: float) -> float:
    """Monotone cosine interpolation from start (step=0) to end (step=total)."""
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * step / total))


def linear_warmup(epoch: int, warmup: int, start: float, end: float) -> float:
    if epoch >= warmup:
        return end
    return start + (end - start) * epoch / warmup


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class DINODataModule(L.LightningDataModule):
    def __init__(self, cfg: DINOConfig, data_root: str = "./data"):
        super().__init__()
        self.cfg = cfg
        self.data_root = data_root

    def setup(self, stage: str | None = None):
        self.train_ds = MNISTMultiCrop(self.cfg, train=True,  root=self.data_root)
        self.val_ds   = MNISTMultiCrop(self.cfg, train=False, root=self.data_root)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_multicrop,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_multicrop,
        )


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DINOModule(L.LightningModule):
    def __init__(self, cfg: DINOConfig, knn_samples: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.student, self.teacher = build_student_teacher(cfg)
        self.criterion = DINOLoss(cfg)

        # Mutable state updated by per-epoch / per-step schedules.
        self.teacher_temp     = cfg.temperature.teacher_temp_start
        self.teacher_momentum = cfg.momentum.momentum_start
        self._total_steps     = 1   # overwritten in on_train_start

        # k-NN eval sample size.  Default: 10 % of MNIST val set (10 000 → 1 000).
        # A 1 000×1 000 similarity matrix fits easily in memory and runs in < 1 s.
        self.knn_samples = knn_samples if knn_samples is not None else int(10_000 * 0.10)

        # Buffers for accumulating validation features (cleared each epoch).
        self._val_feats:  list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    # ------------------------------------------------------------------ train

    def on_train_start(self) -> None:
        # Total optimizer steps for teacher EMA cosine schedule (per-step).
        # Paper: "λ following a cosine schedule from 0.996 to 1 during training."
        self._total_steps = int(self.trainer.estimated_stepping_batches)

    def on_train_epoch_start(self) -> None:
        epoch = self.current_epoch
        total = self.cfg.schedule.total_epochs

        # Cosine weight-decay schedule (paper Sec 3 Implementation).
        wd = cosine_schedule(
            epoch, total,
            self.cfg.optimizer.weight_decay_start,
            self.cfg.optimizer.weight_decay_end,
        )
        opt = self.optimizers()
        for pg in opt.param_groups:
            if pg.get("apply_wd", True):
                pg["weight_decay"] = wd
        self.log("train/weight_decay", wd, on_epoch=True, on_step=False)

        # Linear warmup for teacher temperature τ_t (paper: 0.04→0.07, 30 ep).
        self.teacher_temp = linear_warmup(
            epoch,
            self.cfg.temperature.teacher_temp_warmup_epochs,
            self.cfg.temperature.teacher_temp_start,
            self.cfg.temperature.teacher_temp_end,
        )
        self.log("train/teacher_temp", self.teacher_temp, on_epoch=True, on_step=False)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        crops, _ = batch
        # crops: list[n_global + n_local] of (B, 1, H, W)
        n_global = self.cfg.data.n_global_crops

        student_out = [self.student(v) for v in crops]
        with torch.no_grad():
            teacher_out = [self.teacher(v) for v in crops[:n_global]]

        loss = self.criterion(student_out, teacher_out, self.teacher_temp)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # EMA teacher update — done per step, cosine schedule for λ.
        # Paper: "θ_t ← λ·θ_t + (1−λ)·θ_s, λ cosine 0.996→1"
        # Per-step (not per-epoch) is more faithful to the paper.
        self.teacher_momentum = cosine_schedule(
            self.global_step, self._total_steps,
            self.cfg.momentum.momentum_start,
            self.cfg.momentum.momentum_end,
        )
        m = self.teacher_momentum
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.mul_(m).add_((1.0 - m) * ps.data)

        self.log("train/teacher_momentum", m, on_step=True, on_epoch=False)

    # ---------------------------------------------------------------- validate

    def on_validation_epoch_start(self) -> None:
        self._val_feats  = []
        self._val_labels = []

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        crops, labels = batch
        n_global = self.cfg.data.n_global_crops

        # DINO loss on the validation split — monitors representation collapse.
        student_out = [self.student(v) for v in crops]
        teacher_out = [self.teacher(v) for v in crops[:n_global]]
        loss = self.criterion(student_out, teacher_out, self.teacher_temp)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Collect L2-normalised backbone features for k-NN at epoch end.
        # Global crop[0] provides a canonical view (consistent crop size).
        feats = F.normalize(self.student.get_features(crops[0]), dim=-1)
        self._val_feats.append(feats.cpu())
        self._val_labels.append(labels.cpu())

        return loss

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.cfg.eval.knn_eval_every_n_epochs != 0:
            return

        all_feats  = torch.cat(self._val_feats)    # (N_val, D)
        all_labels = torch.cat(self._val_labels)   # (N_val,)

        # Draw a random subsample from the full val set.
        # Default: 1 000 samples → 1 000×1 000 similarity matrix (~4 MB, < 1 s).
        n = min(self.knn_samples, len(all_feats))
        perm   = torch.randperm(len(all_feats))[:n]
        feats  = all_feats[perm]
        labels = all_labels[perm]

        acc = self._knn_classify(feats, labels, k=self.cfg.data.knn_k)
        self.log("val/knn_accuracy",  acc,      prog_bar=True)
        self.log("val/knn_n_samples", float(n), prog_bar=False)

    def _knn_classify(
        self,
        feats:  torch.Tensor,   # (N, D)  L2-normalised val subsample
        labels: torch.Tensor,   # (N,)
        k: int,
    ) -> float:
        """Temperature-weighted k-NN on a subsampled validation set.

        Paper (Appendix F.1, following Wu et al. 2018): votes are weighted by
        exp(cosine_sim / T) so near neighbours dominate.  Prediction is the
        class with the highest total weighted vote (argmax, not majority).
        """
        N   = feats.shape[0]
        T   = self.cfg.data.knn_temperature
        n_classes = int(labels.max().item()) + 1

        sim = feats @ feats.T                           # (N, N)  cosine similarity
        sim.fill_diagonal_(float("-inf"))               # exclude self

        topk_sim, topk_idx = sim.topk(k, dim=1)        # (N, k)
        neighbor_labels = labels[topk_idx]              # (N, k)

        # Temperature-scaled weights.
        weights = (topk_sim / T).exp()                  # (N, k)

        # Weighted vote per class: (N, k, C) → sum over k → (N, C).
        neighbor_one_hot = torch.zeros(N, k, n_classes, device=feats.device)
        neighbor_one_hot.scatter_(2, neighbor_labels.unsqueeze(2), 1.0)
        votes = (weights.unsqueeze(2) * neighbor_one_hot).sum(1)  # (N, C)

        pred = votes.argmax(1)
        return (pred == labels.to(feats.device)).float().mean().item()

    # ------------------------------------------------------------- optimizers

    def configure_optimizers(self):
        # Exclude bias + norm parameters from weight decay — standard for ViT/ResNet.
        # AdamW paper (Loshchilov & Hutter) recommends this split.
        decay, no_decay = [], []
        for name, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            if any(token in name for token in ("bias", ".bn", "norm", "ln")):
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": decay,    "apply_wd": True},
            {"params": no_decay, "apply_wd": False, "weight_decay": 0.0},
        ]

        lr = self.cfg.lr   # base_lr * batch_size / 256  (linear scaling rule)
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=tuple(self.cfg.optimizer.betas),
            weight_decay=self.cfg.optimizer.weight_decay_start,
        )

        warmup_ep  = self.cfg.schedule.warmup_epochs
        total_ep   = self.cfg.schedule.total_epochs
        min_lr     = self.cfg.schedule.min_lr

        # Linear warmup: 1e-4 × lr → lr  over warmup_epochs.
        warmup_sched = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_ep,
        )
        # Cosine decay: lr → min_lr over remaining epochs.
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(total_ep - warmup_ep, 1),
            eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_ep],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DINO on MNIST")
    parser.add_argument(
        "--knn-samples", type=int, default=None, metavar="N",
        help=(
            "Number of validation samples for the k-NN accuracy check "
            "(builds an N×N similarity matrix). "
            "Default: 10 %% of the validation set  (10 000 × 0.10 = 1 000)."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="Override config total_epochs.",
    )
    parser.add_argument(
        "--precision", type=str, default=None,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Override config precision (e.g. '32' for MPS / CPU).",
    )
    args = parser.parse_args()

    # Apply CLI overrides to config.
    cfg = default_cfg
    if args.epochs is not None:
        cfg.schedule.total_epochs = args.epochs
    if args.precision is not None:
        cfg.train.precision = args.precision

    import time

    torch.set_float32_matmul_precision("high")  # faster on Ampere+ GPUs

    model = DINOModule(cfg, knn_samples=args.knn_samples)
    data  = DINODataModule(cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath="dino/checkpoints",
        # metric logged only every knn_eval_every_n_epochs → saves on those epochs only
        filename="ep{epoch:03d}_knn{val/knn_accuracy:.3f}",
        monitor="val/knn_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress   = TQDMProgressBar(refresh_rate=20)

    logger = TensorBoardLogger(save_dir="dino/logs", name="mnist_dino")

    trainer = L.Trainer(
        max_epochs=cfg.schedule.total_epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=[checkpoint_cb, lr_monitor, progress],
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=1,
        enable_model_summary=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=data)
    total_seconds = time.perf_counter() - t0

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available() else 0.0
    )
    num_params_m = sum(p.numel() for p in model.student.parameters()) / 1e6
    best_acc = float(checkpoint_cb.best_model_score or 0.0)

    print("\n---")
    print(f"val_knn_accuracy: {best_acc:.6f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"total_epochs:     {cfg.schedule.total_epochs}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params_M:     {num_params_m:.1f}")
    print(f"knn_samples:      {model.knn_samples}")
