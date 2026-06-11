"""
DINO on MNIST — comprehensive model evaluation.

Loads the best checkpoint, saves weights as best.pth, then produces:
  1. kNN accuracy vs k  (k = 1, 3, 5, 7, 10, 15, 20, 30, 50, 100)
  2. Confidence histogram: correct vs wrong predictions
  3. Confusion matrix
  4. Per-class accuracy bar chart
  5. t-SNE embedding coloured by class
  6. Nearest-neighbour visualisation (query + top-5 neighbours)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from dino.config import cfg
from dino.ds import MNISTClean
from dino.network import build_student_teacher
from dino.train import DINOModule

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT = "dino/checkpoints/epepoch=008_knnval/knn_accuracy=0.983.ckpt"
BEST_PTH = "dino/best.pth"
OUT_PNG  = "dino/evaluation.png"

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading checkpoint …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module = DINOModule.load_from_checkpoint(CKPT, cfg=cfg, strict=False,
                                          map_location=DEVICE)
student = module.student.eval().to(DEVICE)

# Save backbone weights as best.pth
torch.save(student.state_dict(), BEST_PTH)
print(f"Saved student weights → {BEST_PTH}")

# ── 2. Extract features (clean, no augmentation) ───────────────────────────────
def get_features(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = MNISTClean(cfg, train=train, root="./data")
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(DEVICE)
            f = F.normalize(student.get_features(imgs), dim=-1)
            feats.append(f.cpu())
            labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)

print("Extracting train features (bank) …")
bank_feats,  bank_labels  = get_features(train=True)   # (60000, 512)
print("Extracting val features (query) …")
query_feats, query_labels = get_features(train=False)  # (10000, 512)

# Full similarity matrix — computed once, reused for all k
print("Computing 10k×60k similarity matrix …")
sim = query_feats @ bank_feats.T   # (10000, 60000)

# ── 3. kNN at multiple k values ────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50, 100]
T = cfg.data.knn_temperature   # 0.06
N_q = query_feats.shape[0]
n_classes = 10

def knn_classify(k):
    topk_sim, topk_idx = sim.topk(k, dim=1)          # (N_q, k)
    nbr_labels = bank_labels[topk_idx]                # (N_q, k)
    weights    = (topk_sim / T).exp()                 # (N_q, k)
    one_hot    = torch.zeros(N_q, k, n_classes)
    one_hot.scatter_(2, nbr_labels.unsqueeze(2), 1.0)
    votes = (weights.unsqueeze(2) * one_hot).sum(1)   # (N_q, C)
    preds = votes.argmax(1)
    correct = (preds == query_labels)
    acc = correct.float().mean().item()
    # Confidence = winning vote fraction (max vote / total vote)
    conf = votes.max(1).values / votes.sum(1)         # (N_q,)
    return acc, preds, correct, conf

print("Running kNN sweep …")
k_accs = []
for k in K_VALUES:
    acc, _, _, _ = knn_classify(k)
    k_accs.append(acc * 100)
    print(f"  k={k:3d}  →  {acc*100:.2f}%")

# Best k results (k=15 as configured)
_, best_preds, best_correct, best_conf = knn_classify(k=15)

# ── 4. Per-class accuracy ──────────────────────────────────────────────────────
per_class_acc = []
for c in range(n_classes):
    mask = query_labels == c
    per_class_acc.append(best_correct[mask].float().mean().item() * 100)

# ── 5. Confusion matrix ────────────────────────────────────────────────────────
cm = confusion_matrix(query_labels.numpy(), best_preds.numpy())
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

# ── 6. t-SNE (on 3000 val samples for speed) ──────────────────────────────────
print("Computing t-SNE (3000 samples) …")
rng  = np.random.default_rng(42)
idx  = rng.choice(N_q, size=3000, replace=False)
tsne = TSNE(n_components=2, perplexity=40, max_iter=800, random_state=42)
emb  = tsne.fit_transform(query_feats[idx].numpy())
tsne_labels = query_labels[idx].numpy()

# ── 7. Nearest-neighbour visualisation ────────────────────────────────────────
# For each class, pick a representative query and show its top-5 neighbours
print("Preparing nearest-neighbour examples …")
SHOW_CLASSES = list(range(10))
# Use clean PIL images (from base MNIST) for display
from torchvision.datasets import MNIST
from torchvision import transforms
base_train = MNIST("./data", train=True,  download=True)
base_val   = MNIST("./data", train=False, download=True)

# For each class in val, pick the query closest to the class centroid
nn_examples = []   # list of (query_idx, [bank_idx_top5])
for c in SHOW_CLASSES:
    q_mask = query_labels == c
    q_idx  = torch.where(q_mask)[0]
    centroid = F.normalize(query_feats[q_mask].mean(0, keepdim=True), dim=-1)
    dists = (query_feats[q_idx] * centroid).sum(1)
    rep_local = dists.argmax().item()
    rep_global = q_idx[rep_local].item()
    # top-5 neighbours from bank
    nbr_sim = sim[rep_global]
    top5 = nbr_sim.topk(5).indices.tolist()
    nn_examples.append((rep_global, top5))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Building figure …")

BG       = "#0d1117"
PANEL_BG = "#111827"
GRID     = "#1e2a3a"
WHITE    = "#e8eaf0"
DIM      = "#5a6a7a"

CLASS_COLORS = [
    "#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
    "#3498db","#9b59b6","#e91e63","#00bcd4","#ff5722",
]
DIGIT_CMAP = ListedColormap(CLASS_COLORS)

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)
    if title:  ax.set_title(title, color=WHITE, fontsize=9.5, pad=6, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=8.5)

fig = plt.figure(figsize=(20, 18), facecolor=BG)
gs  = gridspec.GridSpec(
    4, 4,
    figure=fig,
    hspace=0.48, wspace=0.38,
    top=0.93, bottom=0.04,
    left=0.05, right=0.97,
    height_ratios=[1.1, 1.1, 1.0, 1.4],
)

# ── Panel A: k sweep ──────────────────────────────────────────────────────────
ax_k = fig.add_subplot(gs[0, :2])
best_k_idx = int(np.argmax(k_accs))

ax_k.plot(K_VALUES, k_accs, color="#5dade2", linewidth=2.2, zorder=3)
ax_k.scatter(K_VALUES, k_accs, color="#5dade2", s=70, zorder=4,
             edgecolors=WHITE, linewidths=0.8)
ax_k.scatter([K_VALUES[best_k_idx]], [k_accs[best_k_idx]],
             color="#f39c12", s=130, zorder=5, edgecolors=WHITE, linewidths=1.0)
for k, acc in zip(K_VALUES, k_accs):
    va = "bottom" if k != K_VALUES[best_k_idx] else "top"
    dy = 0.05 if va == "bottom" else -0.08
    ax_k.text(k, acc + dy, f"{acc:.2f}%", ha="center", va=va,
              fontsize=7.5, color=WHITE)
ax_k.axhline(k_accs[best_k_idx], color="#f39c12", linewidth=0.7,
             linestyle="--", alpha=0.4)
style(ax_k, title="kNN Accuracy vs k  (60k train bank, T=0.06)",
      xlabel="k  (number of neighbours)", ylabel="Accuracy (%)")
ax_k.set_ylim(min(k_accs) - 1.5, max(k_accs) + 1.5)
ax_k.set_xticks(K_VALUES)
ax_k.grid(axis="x", color=GRID, linewidth=0.5, linestyle=":", alpha=0.5)

# ── Panel B: Confidence histogram ────────────────────────────────────────────
ax_conf = fig.add_subplot(gs[0, 2:])

conf_correct = best_conf[best_correct].numpy()
conf_wrong   = best_conf[~best_correct].numpy()

bins = np.linspace(0, 1, 51)
ax_conf.hist(conf_correct, bins=bins, color="#2ecc71", alpha=0.7,
             label=f"Correct  (n={best_correct.sum():,})", density=True)
ax_conf.hist(conf_wrong, bins=bins, color="#e74c3c", alpha=0.8,
             label=f"Wrong  (n={(~best_correct).sum():,})", density=True)

ax_conf.axvline(np.median(conf_correct), color="#2ecc71", linewidth=1.2,
                linestyle="--", alpha=0.7)
ax_conf.axvline(np.median(conf_wrong), color="#e74c3c", linewidth=1.2,
                linestyle="--", alpha=0.7)
ax_conf.text(np.median(conf_correct) + 0.01, ax_conf.get_ylim()[1] * 0.5,
             f"med={np.median(conf_correct):.2f}", color="#2ecc71", fontsize=7.5)
ax_conf.text(np.median(conf_wrong) + 0.01, ax_conf.get_ylim()[1] * 0.8,
             f"med={np.median(conf_wrong):.2f}", color="#e74c3c", fontsize=7.5)

legend = ax_conf.legend(fontsize=8.5, facecolor=PANEL_BG, edgecolor=GRID,
                        labelcolor=WHITE, framealpha=0.9)
style(ax_conf, title="Vote Confidence: Correct vs Wrong Predictions  (k=15)",
      xlabel="Confidence  (winning-class vote fraction)", ylabel="Density")
ax_conf.set_xlim(0, 1)

# ── Panel C: Per-class accuracy ───────────────────────────────────────────────
ax_cls = fig.add_subplot(gs[1, :2])
digit_names = [str(i) for i in range(10)]
bars = ax_cls.bar(digit_names, per_class_acc,
                  color=CLASS_COLORS, edgecolor=PANEL_BG, linewidth=0.5, zorder=3)
for bar, acc in zip(bars, per_class_acc):
    ax_cls.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, color=WHITE)
ax_cls.axhline(np.mean(per_class_acc), color=WHITE, linewidth=0.8,
               linestyle="--", alpha=0.4)
ax_cls.text(9.5, np.mean(per_class_acc) + 0.2, f"avg", ha="right",
            fontsize=7.5, color=DIM)
style(ax_cls, title="Per-Class kNN Accuracy  (k=15)",
      xlabel="Digit class", ylabel="Accuracy (%)")
ax_cls.set_ylim(min(per_class_acc) - 1.5, 101.5)
ax_cls.grid(axis="x", visible=False)

# ── Panel D: Confusion matrix ─────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 2:])
im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax_cm.set_xticks(range(10)); ax_cm.set_yticks(range(10))
ax_cm.set_xticklabels(range(10), color=DIM, fontsize=8)
ax_cm.set_yticklabels(range(10), color=DIM, fontsize=8)
for i in range(10):
    for j in range(10):
        val = cm_norm[i, j]
        color = "white" if val > 0.5 else DIM
        if val > 0.005:
            ax_cm.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=7, color=color)
ax_cm.set_facecolor(PANEL_BG)
ax_cm.set_title("Confusion Matrix  (row-normalised, k=15)",
                color=WHITE, fontsize=9.5, pad=6, fontweight="bold")
ax_cm.set_xlabel("Predicted", color=DIM, fontsize=8.5)
ax_cm.set_ylabel("True", color=DIM, fontsize=8.5)
cb = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=DIM, labelsize=7)
cb.outline.set_edgecolor(GRID)

# ── Panel E: t-SNE ────────────────────────────────────────────────────────────
ax_tsne = fig.add_subplot(gs[2, :2])
for c in range(n_classes):
    mask = tsne_labels == c
    ax_tsne.scatter(emb[mask, 0], emb[mask, 1],
                    color=CLASS_COLORS[c], s=6, alpha=0.65,
                    edgecolors="none", label=str(c), zorder=3)
ax_tsne.legend(title="Digit", fontsize=7.5, title_fontsize=8,
               facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE,
               markerscale=2.5, ncol=5, loc="upper right",
               framealpha=0.9)
ax_tsne.set_facecolor(PANEL_BG)
ax_tsne.set_title("t-SNE of Val Features  (3000 samples, 512-d → 2-d)",
                  color=WHITE, fontsize=9.5, pad=6, fontweight="bold")
ax_tsne.tick_params(colors=DIM, labelsize=7)
for sp in ax_tsne.spines.values(): sp.set_edgecolor(GRID)
ax_tsne.set_xticks([]); ax_tsne.set_yticks([])

# ── Panel F: Nearest neighbours ───────────────────────────────────────────────
ax_nn = fig.add_subplot(gs[2:, 2:])
ax_nn.set_facecolor(PANEL_BG)
ax_nn.set_title("Query → Top-5 Nearest Neighbours from Train Bank",
                color=WHITE, fontsize=9.5, pad=6, fontweight="bold")
ax_nn.axis("off")

to_pil = transforms.ToPILImage()

n_rows = len(SHOW_CLASSES)
n_cols = 6  # query + 5 neighbours
cell_w = 1.0 / n_cols
cell_h = 1.0 / n_rows

for row, (q_idx, nbr_idxs) in enumerate(nn_examples):
    y_top = 1.0 - row * cell_h

    for col, (img_src, is_bank, img_idx) in enumerate(
        [(base_val, False, q_idx)]
        + [(base_train, True, ni) for ni in nbr_idxs]
    ):
        img_pil = img_src[img_idx][0]
        x_left = col * cell_w

        # Image axes inset
        inset = ax_nn.inset_axes(
            [x_left + 0.005, y_top - cell_h + 0.005,
             cell_w - 0.01, cell_h - 0.01],
            transform=ax_nn.transAxes,
        )
        inset.imshow(img_pil, cmap="gray", vmin=0, vmax=255)
        inset.axis("off")

        # Coloured border: class colour for query, grey for neighbours
        true_lbl  = query_labels[q_idx].item() if not is_bank else bank_labels[img_idx].item()
        border_c  = CLASS_COLORS[query_labels[q_idx].item()] if col == 0 else \
                    (CLASS_COLORS[true_lbl] if true_lbl == query_labels[q_idx].item() else "#e74c3c")
        for sp in inset.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border_c)
            sp.set_linewidth(2.0 if col == 0 else 1.2)

        # Label under first column
        if col == 0:
            ax_nn.text(x_left + cell_w / 2, y_top - cell_h + 0.002,
                       f"  '{true_lbl}'", ha="left", va="bottom",
                       fontsize=7, color=CLASS_COLORS[true_lbl],
                       transform=ax_nn.transAxes)

# Column headers
header_labels = ["Query", "NN 1", "NN 2", "NN 3", "NN 4", "NN 5"]
for col, lbl in enumerate(header_labels):
    ax_nn.text((col + 0.5) * cell_w, 1.01, lbl,
               ha="center", va="bottom", fontsize=8,
               color="#5dade2" if col == 0 else DIM,
               transform=ax_nn.transAxes, fontweight="bold" if col == 0 else "normal")

# ── Panel G: Wrong predictions (top confusion pairs, high-confidence first) ───
# Collect wrong prediction metadata
wrong_mask  = ~best_correct
wrong_true  = query_labels[wrong_mask].numpy()
wrong_pred  = best_preds[wrong_mask].numpy()
wrong_conf  = best_conf[wrong_mask].numpy()
wrong_idxs  = torch.where(wrong_mask)[0].numpy()

pair_counts = Counter(zip(wrong_true.tolist(), wrong_pred.tolist()))
top_pairs   = [pair for pair, _ in pair_counts.most_common(5)]

ax_wrong = fig.add_subplot(gs[3, :2])
ax_wrong.set_facecolor(PANEL_BG)
ax_wrong.set_title(
    f"Most Common Wrong Predictions  (k=15, {wrong_mask.sum()} errors total)"
    "  ·  sorted high-confidence first",
    color=WHITE, fontsize=9.5, pad=6, fontweight="bold",
)
ax_wrong.axis("off")

N_SHOW  = 6   # images per pair
LABEL_W = 0.14   # fraction of panel width reserved for row label
cell_w  = (1.0 - LABEL_W) / N_SHOW
cell_h  = 1.0 / len(top_pairs)

for row, (true_c, pred_c) in enumerate(top_pairs):
    mask     = (wrong_true == true_c) & (wrong_pred == pred_c)
    idxs     = wrong_idxs[mask]
    confs    = wrong_conf[mask]
    order    = np.argsort(-confs)[:N_SHOW]   # highest confidence first
    sel_idxs = idxs[order]
    sel_conf = confs[order]

    y_top = 1.0 - row * cell_h

    # Row label: true → pred + count
    ax_wrong.text(
        LABEL_W / 2, y_top - cell_h / 2,
        f"True {true_c} → Pred {pred_c}\n(n={mask.sum()})",
        ha="center", va="center", fontsize=8,
        color=CLASS_COLORS[true_c], transform=ax_wrong.transAxes,
        fontweight="bold",
    )

    for col, (val_idx, conf_val) in enumerate(zip(sel_idxs, sel_conf)):
        img_pil = base_val[int(val_idx)][0]
        x_left  = LABEL_W + col * cell_w

        inset = ax_wrong.inset_axes(
            [x_left + 0.004, y_top - cell_h + 0.01,
             cell_w - 0.008, cell_h - 0.02],
            transform=ax_wrong.transAxes,
        )
        inset.imshow(img_pil, cmap="gray", vmin=0, vmax=255)
        inset.axis("off")

        # Red border — all wrong predictions
        for sp in inset.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#e74c3c")
            sp.set_linewidth(1.8)

        # Confidence below image
        inset.text(0.5, -0.16, f"conf {conf_val:.2f}",
                   ha="center", va="top", fontsize=6.5, color="#e74c3c",
                   transform=inset.transAxes)

# Column headers
ax_wrong.text(LABEL_W / 2, 1.03, "Confusion pair",
              ha="center", va="bottom", fontsize=8, color=DIM,
              transform=ax_wrong.transAxes)
for col in range(N_SHOW):
    x_c = LABEL_W + (col + 0.5) * cell_w
    ax_wrong.text(x_c, 1.03, f"Example {col+1}",
                  ha="center", va="bottom", fontsize=8, color=DIM,
                  transform=ax_wrong.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.965,
         "DINO on MNIST — Model Evaluation  ·  Best checkpoint (98.32% kNN accuracy)",
         ha="center", va="top", fontsize=14, color=WHITE, fontweight="bold")
fig.text(0.5, 0.947,
         "Student backbone  ·  512-d L2-normalised features  ·  10-epoch self-supervised training",
         ha="center", va="top", fontsize=9, color=DIM)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved figure → {OUT_PNG}")
print(f"Saved weights → {BEST_PTH}")
