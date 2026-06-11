"""Generate accuracy progression chart for the DINO autoresearch run."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Each tuple: (exp_x, accuracy, label_text, delta_text, category, text_x, text_y)
# text_x / text_y: position of annotation box in data coordinates
milestones = [
    (1,    5.6,  "Baseline\n(broken eval)",          "",        "bug",       1,    -11),
    (2,   24.6,  "Fix: kNN evaluated\nevery epoch",  "+19.0%",  "bug",       2,     36),
    (3,   31.2,  "Fix LR warmup\n(was too long)",     "+6.6%",   "training",  3,     18),
    (18,  51.7,  "Simpler head\n(2-layer MLP)",        "+7.3%",   "arch",     18,    62),
    (41,  64.0,  "Fix kNN temperature\n(was unused)",  "+12.3%",  "bug",      41,    51),
    (45,  79.2,  "Local crop\n14 → 20 px",             "+10.1%",  "augment",  42,    90),
    (50,  80.7,  "Avg pool\n→ max pool",               "+1.5%",   "arch",     53,    71),
    (53,  87.9,  "Global crop scale\n0.5 → 0.7",       "+7.2%",   "augment",  55,    97),
    (65,  93.3,  "ReLU → SiLU\n(smoother grads)",      "+5.4%",   "arch",     63,    80),
    (69,  94.1,  "Gradient clip\n3.0 → 5.0",           "+0.8%",   "training", 69,   104),
    (77,  95.9,  "Sharper teacher\ntemp 0.04→0.03",    "+1.8%",   "training", 74,    83),
    (84,  96.7,  "Longer warmup\n2→4 epochs",          "+0.8%",   "training", 80,   106),
    (91,  97.3,  "kNN tuning\n(k & temperature)",      "+0.6%",   "eval",     90,    84),
    (102, 98.3,  "Train-bank eval\n(paper protocol)",  "+1.0%",   "eval",    101,   108),
]

COLORS = {
    "bug":      "#e74c3c",
    "arch":     "#5dade2",
    "training": "#2ecc71",
    "augment":  "#f39c12",
    "eval":     "#a569bd",
}
LEGEND_LABELS = {
    "bug":      "Bug fix",
    "arch":     "Architecture",
    "training": "Training",
    "augment":  "Augmentation",
    "eval":     "Evaluation",
}

fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Grid ─────────────────────────────────────────────────────────────────────
for yg in [20, 40, 60, 80, 90, 95]:
    ax.axhline(yg, color="#161625", linewidth=0.8, zorder=0)

# ── Step fill + line ─────────────────────────────────────────────────────────
step_xs, step_ys = [], []
for i, (x, y, *_) in enumerate(milestones):
    if i > 0:
        step_xs.append(x)
        step_ys.append(milestones[i - 1][1])
    step_xs.append(x)
    step_ys.append(y)
ax.fill_between(step_xs, step_ys, alpha=0.07, color="#5dade2", zorder=0)
ax.plot(step_xs, step_ys, color="#ffffff", linewidth=2.0, alpha=0.18, zorder=1)

# ── Dots ──────────────────────────────────────────────────────────────────────
for x, y, label, delta, cat, tx, ty in milestones:
    c = COLORS[cat]
    ax.scatter(x, y, s=170, color=c, zorder=5, edgecolors="white", linewidths=0.9)

# ── Annotations ───────────────────────────────────────────────────────────────
for x, y, label, delta, cat, tx, ty in milestones:
    c = COLORS[cat]
    full = f"{label}\n{delta}" if delta else label

    ax.annotate(
        full,
        xy=(x, y),
        xytext=(tx, ty),
        ha="center", va="center",
        fontsize=8.5, color=c,
        arrowprops=dict(
            arrowstyle="-",
            color=c, alpha=0.45, lw=0.9,
            shrinkA=6, shrinkB=4,
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#0a0e1a",
            edgecolor=c, alpha=0.93,
            linewidth=0.9,
        ),
        zorder=8,
    )

    # Accuracy value just above the dot
    ax.text(x, y + 2.2, f"{y:.1f}%", ha="center", va="bottom",
            fontsize=7.5, color="white", fontweight="bold", zorder=6)

# ── Final dashed baseline ─────────────────────────────────────────────────────
ax.axhline(98.3, color=COLORS["eval"], linestyle="--", linewidth=1.0, alpha=0.3)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-3, 113)
ax.set_ylim(-20, 120)
ax.set_xlabel("Experiment number", color="#888888", fontsize=12, labelpad=8)
ax.set_ylabel("kNN accuracy  (%)", color="#888888", fontsize=12, labelpad=8)
ax.tick_params(colors="#555555", labelsize=9.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e1e2e")
ax.set_yticks([0, 20, 40, 60, 80, 90, 95, 100])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "90%", "95%", "100%"],
                   color="#666666")
ax.set_xticks(range(0, 110, 10))

# Y-axis labels on the right
for yg, label in [(20, "20%"), (40, "40%"), (60, "60%"), (80, "80%"),
                  (90, "90%"), (95, "95%")]:
    ax.text(112, yg, label, va="center", ha="left",
            fontsize=7.5, color="#333333")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COLORS[k], label=LEGEND_LABELS[k])
    for k in ["bug", "arch", "training", "augment", "eval"]
]
leg = ax.legend(
    handles=handles, loc="upper left", framealpha=0.4,
    facecolor="#0a0e1a", edgecolor="#2a2a3a",
    labelcolor="white", fontsize=10,
    title="Change type", title_fontsize=9,
    labelspacing=0.6, borderpad=0.9,
)
leg.get_title().set_color("#aaaaaa")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "DINO on MNIST  ·  104 experiments  ·  5.6% → 98.3% kNN accuracy  ·  10 epochs, no labels",
    color="white", fontsize=13.5, pad=14, fontweight="bold",
)

plt.tight_layout(pad=1.5)
out = "dino/accuracy_progress.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
