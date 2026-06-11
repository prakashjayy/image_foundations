"""
Dashboard: DINO on MNIST — best run (98.32% kNN accuracy, 10 epochs)
Reads TensorBoard events from version_106 and produces a single figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Load data ─────────────────────────────────────────────────────────────────
LOG_DIR = "dino/logs/mnist_dino/version_106"
ea = EventAccumulator(LOG_DIR)
ea.Reload()

def scalars(tag):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals  = np.array([e.value for e in events])
    return steps, vals

STEPS_PER_EPOCH = 234   # 60 000 / 256 (drop_last=True)

def to_epoch(steps):
    return steps / STEPS_PER_EPOCH

# Per-epoch (logged at end of each epoch, step = last step of that epoch)
ep_steps, knn_acc   = scalars("val/knn_accuracy")
_,         val_loss  = scalars("val/loss")
_,         tr_loss   = scalars("train/loss_epoch")
_,         wd        = scalars("train/weight_decay")
_,         t_temp    = scalars("train/teacher_temp")

knn_epochs  = to_epoch(ep_steps)           # 1..10
loss_epochs = knn_epochs

# Per-step (logged every log_every_n_steps=50)
step_steps,  tr_loss_step  = scalars("train/loss_step")
_,           momentum_step = scalars("train/teacher_momentum")
step_epochs = to_epoch(step_steps)

# Learning rate (logged per epoch by LearningRateMonitor)
lr_steps, lr_vals = scalars("lr-AdamW/pg1")
lr_epochs = to_epoch(lr_steps)

# ── Style ──────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#111827"
GRID     = "#1e2a3a"
WHITE    = "#e8eaf0"
DIM      = "#5a6a7a"
ACCENT1  = "#5dade2"   # kNN / main blue
ACCENT2  = "#2ecc71"   # training green
ACCENT3  = "#e74c3c"   # val / red
ACCENT4  = "#f39c12"   # orange (LR / teacher temp)
ACCENT5  = "#a569bd"   # purple (momentum)
ACCENT6  = "#48c9b0"   # teal (weight decay)

def style_ax(ax, title="", xlabel="", ylabel="", yticks=None):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=DIM, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="--")
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle=":", alpha=0.5)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, pad=7, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8.5)
    if yticks:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1], color=DIM)
    ax.set_xlim(0, 10.4)
    ax.set_xticks(range(0, 11, 2))
    ax.xaxis.label.set_color(DIM)

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.50,
    wspace=0.38,
    top=0.91, bottom=0.07,
    left=0.06, right=0.97,
    height_ratios=[2.6, 1.4, 1.0],
)

# Row 0: kNN accuracy (col 0-1) + Loss (col 2-3)
ax_knn  = fig.add_subplot(gs[0, :2])
ax_loss = fig.add_subplot(gs[0, 2:])

# Row 1: LR (col 0-1) + Teacher temp (col 2-3)
ax_lr   = fig.add_subplot(gs[1, :2])
ax_temp = fig.add_subplot(gs[1, 2:])

# Row 2: Momentum (col 0-1) + Weight Decay (col 2-3)
ax_mom  = fig.add_subplot(gs[2, :2])
ax_wd   = fig.add_subplot(gs[2, 2:])

# ══════════════════════════════════════════════════════════════════════════════
# 1. kNN Accuracy  (main panel)
# ══════════════════════════════════════════════════════════════════════════════
best_epoch_idx = np.argmax(knn_acc)
best_epoch     = knn_epochs[best_epoch_idx]
best_val       = knn_acc[best_epoch_idx]

ax_knn.fill_between(knn_epochs, knn_acc * 100, alpha=0.12, color=ACCENT1)
ax_knn.plot(knn_epochs, knn_acc * 100, color=ACCENT1, linewidth=2.5, zorder=4)
ax_knn.scatter(knn_epochs, knn_acc * 100, color=ACCENT1, s=60, zorder=5,
               edgecolors=WHITE, linewidths=0.7)

# Annotate each epoch value
for ep, acc in zip(knn_epochs, knn_acc):
    offset = 0.35 if ep != best_epoch else 0.45
    ax_knn.text(ep, acc * 100 + offset, f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=8, color=WHITE, zorder=6)

# Best epoch marker
ax_knn.axvline(best_epoch, color=ACCENT1, linestyle="--", linewidth=1.0, alpha=0.4)
ax_knn.annotate(
    f"  Best: {best_val*100:.2f}%\n  (epoch {int(best_epoch)})",
    xy=(best_epoch, best_val * 100),
    xytext=(best_epoch + 0.3, best_val * 100 - 3.5),
    fontsize=9, color=ACCENT1, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ACCENT1, lw=1.2),
)

# Warmup region shading
ax_knn.axvspan(0, 4, color=ACCENT4, alpha=0.04)
ax_knn.text(2, 79.5, "LR warmup", ha="center", va="bottom",
            fontsize=8, color=ACCENT4, alpha=0.7)

# Big jump annotation (epoch 1→2)
ax_knn.annotate(
    "+15.4pp",
    xy=(2, knn_acc[1] * 100), xytext=(2.5, 88),
    fontsize=8.5, color="#f39c12", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.0),
)

style_ax(ax_knn,
         title="kNN Accuracy  (60k train bank → 10k val queries)",
         ylabel="kNN Accuracy (%)",
         yticks=([75, 80, 85, 90, 93, 96, 98, 100],
                 ["75%", "80%", "85%", "90%", "93%", "96%", "98%", "100%"]))
ax_knn.set_ylim(74, 102)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Loss  (train + val)
# ══════════════════════════════════════════════════════════════════════════════
# Per-step train loss (faint)
ax_loss.plot(step_epochs, tr_loss_step, color=ACCENT2, linewidth=0.9,
             alpha=0.25, zorder=2, label="_nolegend_")
# Per-epoch train loss (solid)
ax_loss.plot(loss_epochs, tr_loss, color=ACCENT2, linewidth=2.2,
             zorder=4, label="Train loss")
ax_loss.scatter(loss_epochs, tr_loss, color=ACCENT2, s=45, zorder=5,
                edgecolors=WHITE, linewidths=0.6)
# Val loss
ax_loss.plot(loss_epochs, val_loss, color=ACCENT3, linewidth=2.2,
             zorder=4, label="Val loss", linestyle="--")
ax_loss.scatter(loss_epochs, val_loss, color=ACCENT3, s=45, zorder=5,
                edgecolors=WHITE, linewidths=0.6)

# Convergence annotation
ax_loss.axhline(2.3, color=GRID, linewidth=0.7, linestyle=":")
ax_loss.text(10.3, 2.3, "≈2.3", va="center", ha="left",
             fontsize=7.5, color=DIM)

legend = ax_loss.legend(fontsize=8.5, facecolor=PANEL_BG,
                         edgecolor=GRID, labelcolor=WHITE,
                         loc="upper right", framealpha=0.9)

style_ax(ax_loss, title="DINO Loss (Cross-Entropy)",
         ylabel="Loss", xlabel="Epoch")
ax_loss.set_ylim(1.5, 8.0)
ax_loss.set_yticks([2, 3, 4, 5, 6, 7])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Learning Rate
# ══════════════════════════════════════════════════════════════════════════════
ax_lr.fill_between(lr_epochs, lr_vals * 1000, alpha=0.10, color=ACCENT4)
ax_lr.plot(lr_epochs, lr_vals * 1000, color=ACCENT4, linewidth=2.2)
ax_lr.scatter(lr_epochs, lr_vals * 1000, color=ACCENT4, s=40, zorder=5,
              edgecolors=WHITE, linewidths=0.6)

ax_lr.axvspan(0, 4, color=ACCENT4, alpha=0.06)
ax_lr.text(2, 0.47, "warmup", ha="center", fontsize=8, color=ACCENT4, alpha=0.8)
ax_lr.axvline(4, color=ACCENT4, linewidth=0.8, linestyle="--", alpha=0.4)
ax_lr.text(4.1, 0.47, "cosine decay →", ha="left", fontsize=8, color=DIM)

style_ax(ax_lr, title="Learning Rate (×10⁻³)",
         ylabel="LR (×10⁻³)", xlabel="Epoch")
ax_lr.set_ylim(-0.02, 0.55)
ax_lr.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax_lr.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"], color=DIM)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Teacher Temperature
# ══════════════════════════════════════════════════════════════════════════════
ax_temp.fill_between(knn_epochs, t_temp, alpha=0.10, color=ACCENT3)
ax_temp.plot(knn_epochs, t_temp, color=ACCENT3, linewidth=2.2)
ax_temp.scatter(knn_epochs, t_temp, color=ACCENT3, s=40, zorder=5,
                edgecolors=WHITE, linewidths=0.6)

# Show where it's headed (projected to epoch 15)
proj_ep  = np.array([10, 15])
proj_val = np.array([0.054, 0.070])
ax_temp.plot(proj_ep, proj_val, color=ACCENT3, linewidth=1.0,
             linestyle=":", alpha=0.4)
ax_temp.axhline(0.070, color=ACCENT3, linewidth=0.6, linestyle="--", alpha=0.3)
ax_temp.text(10.3, 0.0705, "target: 0.07\n(epoch 15)", ha="left", va="center",
             fontsize=7, color=ACCENT3, alpha=0.6)

style_ax(ax_temp, title="Teacher Temperature  (linear warmup)",
         ylabel="τ_teacher", xlabel="Epoch")
ax_temp.set_ylim(0.025, 0.078)
ax_temp.set_yticks([0.03, 0.04, 0.05, 0.06, 0.07])
ax_temp.set_yticklabels(["0.030", "0.040", "0.050", "0.060", "0.070"], color=DIM)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Teacher EMA Momentum
# ══════════════════════════════════════════════════════════════════════════════
ax_mom.plot(step_epochs, momentum_step, color=ACCENT5, linewidth=2.2)
ax_mom.fill_between(step_epochs, momentum_step, alpha=0.10, color=ACCENT5)
ax_mom.axhline(1.0, color=ACCENT5, linewidth=0.7, linestyle="--", alpha=0.3)
ax_mom.text(10.3, 1.0005, "1.000", ha="left", va="center",
            fontsize=7.5, color=ACCENT5, alpha=0.6)

style_ax(ax_mom, title="Teacher EMA Momentum  (cosine → 1.0)",
         ylabel="λ (momentum)", xlabel="Epoch")
ax_mom.set_ylim(0.9958, 1.0008)
ax_mom.set_yticks([0.996, 0.997, 0.998, 0.999, 1.000])
ax_mom.set_yticklabels(["0.996", "0.997", "0.998", "0.999", "1.000"], color=DIM)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Weight Decay
# ══════════════════════════════════════════════════════════════════════════════
ax_wd.fill_between(knn_epochs, wd, alpha=0.10, color=ACCENT6)
ax_wd.plot(knn_epochs, wd, color=ACCENT6, linewidth=2.2)
ax_wd.scatter(knn_epochs, wd, color=ACCENT6, s=40, zorder=5,
              edgecolors=WHITE, linewidths=0.6)
ax_wd.axhline(0.4, color=ACCENT6, linewidth=0.6, linestyle="--", alpha=0.3)
ax_wd.text(10.3, 0.40, "max: 0.4", ha="left", va="center",
           fontsize=7.5, color=ACCENT6, alpha=0.6)

style_ax(ax_wd, title="Weight Decay  (cosine schedule)",
         ylabel="Weight decay", xlabel="Epoch")
ax_wd.set_ylim(-0.01, 0.44)
ax_wd.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

# ══════════════════════════════════════════════════════════════════════════════
# Shared x-tick labels on bottom row only, hide on others
# ══════════════════════════════════════════════════════════════════════════════
for ax in [ax_knn, ax_loss, ax_lr, ax_temp]:
    ax.tick_params(labelbottom=False)
for ax in [ax_mom, ax_wd]:
    ax.tick_params(labelbottom=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main title
# ══════════════════════════════════════════════════════════════════════════════
fig.text(
    0.5, 0.965,
    "DINO on MNIST — Best Run Dashboard  ·  98.32% kNN accuracy  ·  10 epochs, no labels",
    ha="center", va="top", fontsize=14, color=WHITE, fontweight="bold",
)
fig.text(
    0.5, 0.945,
    "ResNet backbone (SiLU, max pool)  ·  2-layer GELU head  ·  6 local crops (20px)  ·  teacher temp warmup 0.03→0.07",
    ha="center", va="top", fontsize=9, color=DIM,
)

out = "dino/training_dashboard.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
