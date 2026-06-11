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
from dino.ds import MNISTClean, MNISTMultiCrop, collate_multicrop
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

    def _clean_loader(self, train: bool) -> DataLoader:
        ds = MNISTClean(self.cfg, train=train, root=self.data_root)
        return DataLoader(
            ds,
            batch_size=self.cfg.train.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
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

        # k-NN eval sample size.  Default: full MNIST val set (10 000).
        self.knn_samples = knn_samples if knn_samples is not None else 10_000

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
        pass

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        crops, labels = batch
        n_global = self.cfg.data.n_global_crops

        # DINO loss on the validation split — monitors representation collapse.
        student_out = [self.student(v) for v in crops]
        teacher_out = [self.teacher(v) for v in crops[:n_global]]
        loss = self.criterion(student_out, teacher_out, self.teacher_temp)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _collect_features(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        feats, labels = [], []
        was_training = self.student.training
        self.student.eval()
        for imgs, lbls in loader:
            imgs = imgs.to(self.device)
            f = F.normalize(self.student.get_features(imgs), dim=-1)
            feats.append(f.cpu())
            labels.append(lbls)
        if was_training:
            self.student.train()
        return torch.cat(feats), torch.cat(labels)

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.cfg.eval.knn_eval_every_n_epochs != 0:
            return

        dm = self.trainer.datamodule
        bank_feats, bank_labels = self._collect_features(dm._clean_loader(train=True))
        query_feats, query_labels = self._collect_features(dm._clean_loader(train=False))

        n = min(self.knn_samples, len(query_feats))
        perm = torch.randperm(len(query_feats))[:n]
        query_feats  = query_feats[perm]
        query_labels = query_labels[perm]

        acc = self._knn_classify(query_feats, query_labels, bank_feats, bank_labels)
        self.log("val/knn_accuracy",  acc,      prog_bar=True)
        self.log("val/knn_n_samples", float(n), prog_bar=False)

    def _knn_classify(
        self,
        query_feats:  torch.Tensor,   # (N_q, D)  L2-normalised query set
        query_labels: torch.Tensor,   # (N_q,)
        bank_feats:   torch.Tensor,   # (N_b, D)  L2-normalised feature bank
        bank_labels:  torch.Tensor,   # (N_b,)
    ) -> float:
        """Temperature-weighted k-NN: train-set bank, val-set queries.

        Paper (Appendix F.1, following Wu et al. 2018): votes are weighted by
        exp(cosine_sim / T) so near neighbours dominate.  Prediction is the
        class with the highest total weighted vote (argmax, not majority).
        """
        k = self.cfg.data.knn_k
        T = self.cfg.data.knn_temperature
        N_q = query_feats.shape[0]
        n_classes = int(bank_labels.max().item()) + 1

        sim = query_feats @ bank_feats.T              # (N_q, N_b) cosine similarity
        topk_sim, topk_idx = sim.topk(k, dim=1)      # (N_q, k)
        neighbor_labels = bank_labels[topk_idx]       # (N_q, k)

        weights = (topk_sim / T).exp()                # (N_q, k)

        neighbor_one_hot = torch.zeros(N_q, k, n_classes, device=query_feats.device)
        neighbor_one_hot.scatter_(2, neighbor_labels.unsqueeze(2), 1.0)
        votes = (weights.unsqueeze(2) * neighbor_one_hot).sum(1)  # (N_q, C)

        pred = votes.argmax(1)
        return (pred == query_labels.to(query_feats.device)).float().mean().item()

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
