Implemented DINO (self-supervised vision transformer), kept it on auto-research mode overnight to find the optimal architecture and training parameters across 104 experiments — no labels used at any point.

![Accuracy progress across 104 experiments](accuracy_progress.png)

A kNN classifier on the frozen backbone features achieved **98.6% accuracy** (k=5), with the configured k=15 reaching 98.3%. The model never saw a single label during training — only raw images.

The training budget is just **10 epochs**. The auto-research loop committed each config change, trained, and rolled back anything that didn't improve validation accuracy.

---

The biggest gains came from four categories of decisions:

**Bug fixes (+31% combined)**
- Fixed kNN not evaluating every epoch — the loop was flying blind without this (+19%)
- Fixed kNN temperature wired to a variable that was never used in the classifier (+12%)

**Architecture (+14% combined)**
- Switched from avg pool → max pool in the backbone — max pool preserves the strongest spatial activation rather than averaging it away (+1.5%)
- Replaced ReLU with SiLU throughout — smoother gradient flow improved convergence significantly (+5.4%)
- Reduced projection head from 3-layer to 2-layer MLP — simpler head generalised better on a small dataset (+7.3%)

**Augmentation (+17% combined)**
- Enlarged local crops from 14 → 20 px and raised global crop scale from 0.5 → 0.7 — the student now sees most of the digit per view, forcing it to learn global structure rather than isolated strokes

**Evaluation protocol (+1%)**
- Switched kNN eval from val-to-val (leave-one-out) to train-bank protocol — using the full 60k clean training set as the retrieval bank is what the DINO paper actually specifies

---

![Training dashboard — best run metrics](training_dashboard.png)

![Model evaluation — kNN sweep, confusion matrix, t-SNE, wrong predictions](evaluation.png)
