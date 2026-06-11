"""
dino/prototype_analysis.py

Analyses the 512-d backbone feature vectors to understand what the model
has learned to represent without any labels.

Panels
------
A  Class-feature activation heatmap  (10 x 512)
     Features sorted by discriminability (most class-specific first).
     Colour = z-scored mean activation per class.
     Diagonal-block structure proves specific features fire per class.

B  Feature discriminability (F-statistic distribution)
     For each of the 512 features: between-class variance / within-class
     variance.  High F = feature cleanly separates at least one class.

C  Feature value distribution
     Histogram of raw backbone values across all 10k images × 512 dims.
     Shows whether the backbone maintains a healthy, spread distribution
     (not collapsed to zero, not saturated).

D  Class-class cosine similarity  (10 x 10)
     Cosine similarity between per-class mean feature vectors.
     Off-diagonal hot spots directly predict which classes get confused.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from dino.config import cfg
from dino.ds import MNISTClean
from dino.train import DINOModule

# ── Config ────────────────────────────────────────────────────────────────────
CKPT    = "dino/checkpoints/epepoch=008_knnval/knn_accuracy=0.983.ckpt"
OUT_PNG = "dino/prototype_analysis.png"
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 10
N_FEAT    = cfg.backbone.feat_dim   # 512

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading checkpoint …")
module  = DINOModule.load_from_checkpoint(CKPT, cfg=cfg, strict=False,
                                           map_location=DEVICE)
student = module.student.eval().to(DEVICE)

# ── Extract raw backbone features for all 10k val images ─────────────────────
print("Extracting backbone features …")
ds     = MNISTClean(cfg, train=False, root="./data")
loader = DataLoader(ds, batch_size=512, shuffle=False,
                    num_workers=4, pin_memory=True)

all_feats  = []
all_labels = []
with torch.no_grad():
    for imgs, lbs in loader:
        f = student.get_features(imgs.to(DEVICE))   # (B, 512) raw backbone output
        all_feats.append(f.cpu())
        all_labels.append(lbs)

feats  = torch.cat(all_feats).numpy()    # (10000, 512)  raw, no L2 norm
labels = torch.cat(all_labels).numpy()  # (10000,)

# ── Per-class mean & std ──────────────────────────────────────────────────────
mean_per_class = np.stack([feats[labels == c].mean(axis=0)
                            for c in range(N_CLASSES)])   # (10, 512)
std_per_class  = np.stack([feats[labels == c].std(axis=0)
                            for c in range(N_CLASSES)])   # (10, 512)

# ── Feature discriminability: one-way F-statistic ────────────────────────────
# F = between-class variance / mean within-class variance
grand_mean      = feats.mean(axis=0)                           # (512,)
between_var     = np.mean((mean_per_class - grand_mean) ** 2,
                           axis=0)                             # (512,)
within_var      = std_per_class.mean(axis=0) ** 2 + 1e-9       # (512,)
f_stat          = between_var / within_var                     # (512,)

# Sort features by F-statistic (most discriminative first)
sort_idx        = np.argsort(-f_stat)
sorted_means    = mean_per_class[:, sort_idx]                  # (10, 512) reordered

# Z-score each feature column so all 512 are on the same colour scale
col_mean = sorted_means.mean(axis=0)
col_std  = sorted_means.std(axis=0) + 1e-9
heatmap  = (sorted_means - col_mean) / col_std                 # (10, 512) z-scored

# ── Class–class cosine similarity (on L2-normalised features) ─────────────────
norm_means  = mean_per_class / (np.linalg.norm(mean_per_class, axis=1,
                                                keepdims=True) + 1e-9)
class_sim   = norm_means @ norm_means.T   # (10, 10)

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#111827"
GRID     = "#1e2a3a"
WHITE    = "#e8eaf0"
DIM      = "#5a6a7a"
CLASS_COLORS = [
    "#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
    "#3498db","#9b59b6","#e91e63","#00bcd4","#ff5722",
]

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", alpha=0.6)
    if title:  ax.set_title(title, color=WHITE, fontsize=10, pad=7, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=8.5)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.50, wspace=0.40,
    top=0.93, bottom=0.04,
    left=0.05, right=0.97,
    height_ratios=[1.1, 1.0, 1.0],
)

# ══════════════════════════════════════════════════════════════════════════════
# A. Class-feature activation heatmap — full width
# ══════════════════════════════════════════════════════════════════════════════
ax_hm = fig.add_subplot(gs[0, :])

im = ax_hm.imshow(
    heatmap,
    aspect="auto",
    cmap="RdBu_r",        # red = excited, blue = suppressed
    vmin=-2.5, vmax=2.5,
    interpolation="nearest",
)
ax_hm.set_facecolor(PANEL_BG)
ax_hm.set_yticks(range(N_CLASSES))
ax_hm.set_yticklabels([f"Digit {c}" for c in range(N_CLASSES)],
                       color=WHITE, fontsize=8.5)
ax_hm.set_xlabel(
    "Feature index  (sorted left→right by discriminability — most class-specific first)",
    color=DIM, fontsize=8.5,
)
ax_hm.tick_params(bottom=True, colors=DIM, labelsize=8)

# Annotate top-5 most discriminative features per class
for c in range(N_CLASSES):
    # Find features where this class has the highest z-score
    class_row = heatmap[c]
    top5 = np.argsort(-class_row)[:3]
    for rank, feat_pos in enumerate(top5):
        ax_hm.plot(feat_pos, c, marker="^", color=CLASS_COLORS[c],
                   markersize=4, zorder=5, alpha=0.85)

cb = plt.colorbar(im, ax=ax_hm, fraction=0.015, pad=0.01)
cb.ax.tick_params(colors=DIM, labelsize=7)
cb.outline.set_edgecolor(GRID)
cb.set_label("z-score  (red = excited for this class,  blue = suppressed)",
             color=DIM, fontsize=7.5)

# Vertical line at the "knee" of the F-statistic curve
n_high_f = (f_stat > np.percentile(f_stat, 75)).sum()
ax_hm.axvline(n_high_f - 0.5, color=WHITE, linewidth=0.8,
               linestyle=":", alpha=0.4)
ax_hm.text(n_high_f, -0.6,
           f"← top-25% discriminative\n   ({n_high_f} features)",
           ha="left", va="bottom", fontsize=7, color=DIM,
           transform=ax_hm.get_xaxis_transform())

ax_hm.set_title(
    "Backbone Feature Activation Map  (512 features × 10 classes, z-scored mean activation)"
    "  ·  triangles = top-3 most excited features per class",
    color=WHITE, fontsize=10, pad=10, fontweight="bold",
)

# ══════════════════════════════════════════════════════════════════════════════
# B. Feature discriminability (F-statistic distribution)
# ══════════════════════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(gs[1, :2])

ax_f.hist(f_stat, bins=60, color="#5dade2", alpha=0.85,
          edgecolor=BG, linewidth=0.3, zorder=3)

p75 = np.percentile(f_stat, 75)
p90 = np.percentile(f_stat, 90)
ax_f.axvline(p75, color="#f39c12", linewidth=1.2, linestyle="--", alpha=0.8)
ax_f.axvline(p90, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.8)
y_max = ax_f.get_ylim()[1]
ax_f.text(p75 + 0.002, y_max * 0.85, f"p75={p75:.3f}", color="#f39c12", fontsize=7.5)
ax_f.text(p90 + 0.002, y_max * 0.65, f"p90={p90:.3f}", color="#e74c3c", fontsize=7.5)

# Annotate percentage of "highly discriminative" features
frac_useful = (f_stat > 0.01).mean() * 100
ax_f.text(0.97, 0.95,
          f"{frac_useful:.0f}% of features have\nF-stat > 0.01",
          ha="right", va="top", fontsize=8.5, color=WHITE,
          transform=ax_f.transAxes,
          bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                    edgecolor="#5dade2", alpha=0.85))

style(ax_f,
      title="Feature Discriminability  (F-statistic: between-class / within-class variance)",
      xlabel="F-statistic  (higher = feature separates classes more cleanly)",
      ylabel="Number of features")

# ══════════════════════════════════════════════════════════════════════════════
# C. Feature value distribution
# ══════════════════════════════════════════════════════════════════════════════
ax_dist = fig.add_subplot(gs[1, 2:])

# Overall distribution across all features × images
all_vals = feats.ravel()
ax_dist.hist(all_vals, bins=100, color="#2ecc71", alpha=0.55,
             density=True, label="All features", zorder=2)

# Overlay per-class distribution for 3 contrasting classes
for c, alpha in [(0, 0.8), (4, 0.8), (9, 0.8)]:
    ax_dist.hist(feats[labels == c].ravel(), bins=100,
                 color=CLASS_COLORS[c], alpha=alpha,
                 density=True, histtype="step", linewidth=1.4,
                 label=f"Digit {c}", zorder=3)

ax_dist.axvline(0, color=WHITE, linewidth=0.8, linestyle=":", alpha=0.4)
ax_dist.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID,
               labelcolor=WHITE, framealpha=0.9, loc="upper right")

style(ax_dist,
      title="Raw Backbone Feature Value Distribution  (all 512 dims × 10k images)",
      xlabel="Feature value  (raw, before L2 normalisation)",
      ylabel="Density")
ax_dist.set_xlim(np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9))

# ══════════════════════════════════════════════════════════════════════════════
# D. Per-class mean feature profile  (top-40 most discriminative features)
# ══════════════════════════════════════════════════════════════════════════════
ax_prof = fig.add_subplot(gs[2, :2])

TOP_K = 40
top_feat_idx = sort_idx[:TOP_K]
x = np.arange(TOP_K)
width = 0.08

for c in range(N_CLASSES):
    vals = mean_per_class[c, top_feat_idx]
    # normalise to [0,1] range for each feature so all are comparable
    ax_prof.bar(x + c * width, vals, width=width,
                color=CLASS_COLORS[c], alpha=0.85, label=str(c))

ax_prof.axhline(0, color=GRID, linewidth=0.6)
ax_prof.legend(title="Digit", fontsize=7, title_fontsize=7.5,
               facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE,
               ncol=10, loc="upper right", framealpha=0.9,
               columnspacing=0.5, handlelength=0.8)
ax_prof.set_xticks(x + 0.5 * width * (N_CLASSES - 1))
ax_prof.set_xticklabels([f"f{sort_idx[i]}" for i in range(TOP_K)],
                         rotation=60, ha="right", fontsize=6, color=DIM)
style(ax_prof,
      title=f"Mean Activation — Top-{TOP_K} Most Discriminative Features",
      xlabel="Feature (original index)",
      ylabel="Mean raw activation")
ax_prof.grid(axis="x", visible=False)

# ══════════════════════════════════════════════════════════════════════════════
# E. Class–class cosine similarity
# ══════════════════════════════════════════════════════════════════════════════
ax_sim = fig.add_subplot(gs[2, 2:])

im2 = ax_sim.imshow(class_sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0,
                    aspect="auto", interpolation="nearest")
ax_sim.set_xticks(range(N_CLASSES))
ax_sim.set_yticks(range(N_CLASSES))
ax_sim.set_xticklabels(range(N_CLASSES), color=DIM, fontsize=8.5)
ax_sim.set_yticklabels(range(N_CLASSES), color=DIM, fontsize=8.5)
ax_sim.set_facecolor(PANEL_BG)
for sp in ax_sim.spines.values(): sp.set_edgecolor(GRID)

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        val   = class_sim[i, j]
        color = "black" if val > 0.6 else WHITE
        ax_sim.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

cb2 = plt.colorbar(im2, ax=ax_sim, fraction=0.046, pad=0.04)
cb2.ax.tick_params(colors=DIM, labelsize=7)
cb2.outline.set_edgecolor(GRID)

ax_sim.set_title("Class–Class Feature Similarity  (cosine of mean backbone vectors)\n"
                 "high off-diagonal = shared feature space = confusion pairs",
                 color=WHITE, fontsize=10, pad=7, fontweight="bold")
ax_sim.set_xlabel("Class", color=DIM, fontsize=8.5)
ax_sim.set_ylabel("Class", color=DIM, fontsize=8.5)

# ══════════════════════════════════════════════════════════════════════════════
# Main title + summary stats
# ══════════════════════════════════════════════════════════════════════════════
high_f_pct = (f_stat > p75).mean() * 100
fig.text(0.5, 0.965,
         "DINO Backbone Feature Analysis  ·  512 raw features per image",
         ha="center", va="top", fontsize=14, color=WHITE, fontweight="bold")
fig.text(0.5, 0.947,
         f"Top-25% discriminative features: {n_high_f}  ·  "
         f"feature value range: [{feats.min():.1f}, {feats.max():.1f}]  ·  "
         f"mean |activation|: {np.abs(feats).mean():.2f}",
         ha="center", va="top", fontsize=9, color=DIM)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT_PNG}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\nF-statistic: mean={f_stat.mean():.4f}  "
      f"max={f_stat.max():.4f}  "
      f"frac>0.01: {(f_stat > 0.01).mean()*100:.1f}%")
print(f"Feature values: mean={feats.mean():.3f}  "
      f"std={feats.std():.3f}  "
      f"range=[{feats.min():.2f}, {feats.max():.2f}]")
print(f"Class-sim off-diagonal: mean={class_sim[~np.eye(N_CLASSES,dtype=bool)].mean():.3f}  "
      f"max={class_sim[~np.eye(N_CLASSES,dtype=bool)].max():.3f}")
