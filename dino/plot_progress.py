"""Generate accuracy progression chart for the DINO autoresearch run."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# (exp_x, accuracy, label_text, delta_text, category, text_x, text_y)
# text_x/text_y in data coordinates; right-side labels staggered high/low
milestones = [
    # Early experiments — spread naturally by log scale
    (1,    5.6,  "Baseline\n(broken eval)",            "",        "bug",       1,     -14),
    (2,   24.6,  "Fix: kNN evaluated\nevery epoch",    "+19.0%",  "bug",       2,      40),
    (3,   31.2,  "Fix LR warmup\n(was too long)",       "+6.6%",   "training",  4.5,   17),
    (18,  51.7,  "Simpler head\n(2-layer MLP)",          "+7.3%",   "arch",      9,     63),
    (41,  64.0,  "Fix kNN temperature\n(was unused)",   "+12.3%",  "bug",       25,    52),

    # Right side — high/low staggered, each step 6–9 pts apart
    (45,  79.2,  "Local crop\n14 → 20 px",              "+10.1%",  "augment",   43,    91),   # HIGH-1
    (50,  80.7,  "Avg pool\n→ max pool",                "+1.5%",   "arch",      52,    60),   # LOW-1
    (53,  87.9,  "Global crop scale\n0.5 → 0.7",        "+7.2%",   "augment",   52,    96),   # HIGH-2
    (65,  93.3,  "ReLU → SiLU\n(smoother grads)",       "+5.4%",   "arch",      63,    70),   # LOW-2
    (69,  94.1,  "Gradient clip\n3.0 → 5.0",            "+0.8%",   "training",  70,   108),   # HIGH-3
    (77,  95.9,  "Sharper teacher\ntemp 0.04→0.03",     "+1.8%",   "training",  74,    75),   # LOW-3
    (84,  96.7,  "Longer warmup\n2→4 epochs",           "+0.8%",   "training",  83,   113),   # HIGH-4
    (91,  97.3,  "kNN tuning\n(k & temperature)",       "+0.6%",   "eval",      89,    82),   # LOW-4
    (102, 98.3,  "Train-bank eval\n(paper protocol)",   "+1.0%",   "eval",     103,   118),   # HIGH-5
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

fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xscale("log")

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

# ── Dots + accuracy labels ────────────────────────────────────────────────────
for x, y, label, delta, cat, tx, ty in milestones:
    c = COLORS[cat]
    ax.scatter(x, y, s=170, color=c, zorder=5, edgecolors="white", linewidths=0.9)
    ax.text(x, y + 2.0, f"{y:.1f}%", ha="center", va="bottom",
            fontsize=7.5, color="white", fontweight="bold", zorder=6)

# ── Annotation boxes ──────────────────────────────────────────────────────────
for x, y, label, delta, cat, tx, ty in milestones:
    c = COLORS[cat]
    full = f"{label}\n{delta}" if delta else label
    ax.annotate(
        full,
        xy=(x, y),
        xytext=(tx, ty),
        ha="center", va="center",
        fontsize=8.5, color=c,
        arrowprops=dict(arrowstyle="-", color=c, alpha=0.45, lw=0.9,
                        shrinkA=6, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0a0e1a",
                  edgecolor=c, alpha=0.93, linewidth=0.9),
        zorder=8,
    )

# ── Final dashed baseline ─────────────────────────────────────────────────────
ax.axhline(98.3, color=COLORS["eval"], linestyle="--", linewidth=1.0, alpha=0.3)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.85, 120)
ax.set_ylim(-26, 128)
ax.set_xlabel("Experiment number  (log scale)", color="#888888", fontsize=12, labelpad=8)
ax.set_ylabel("kNN accuracy  (%)", color="#888888", fontsize=12, labelpad=8)
ax.tick_params(colors="#555555", labelsize=9.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e1e2e")

ax.set_xticks([1, 2, 3, 5, 10, 20, 30, 50, 75, 100])
ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "30", "50", "75", "100"],
                   color="#666666")
ax.set_yticks([0, 20, 40, 60, 80, 90, 95, 100])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "90%", "95%", "100%"],
                   color="#666666")
for xv in [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]:
    ax.axvline(xv, color="#1a1a2a", linewidth=0.5, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [mpatches.Patch(color=COLORS[k], label=LEGEND_LABELS[k])
           for k in ["bug", "arch", "training", "augment", "eval"]]
leg = ax.legend(handles=handles, loc="upper left", framealpha=0.4,
                facecolor="#0a0e1a", edgecolor="#2a2a3a",
                labelcolor="white", fontsize=10,
                title="Change type", title_fontsize=9,
                labelspacing=0.6, borderpad=0.9)
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
