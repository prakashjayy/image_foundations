# autoresearch — DINO on MNIST

Autonomous experiment loop for self-supervised representation learning.
Each run trains DINO on MNIST and measures k-NN classification accuracy.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun10`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `dino/config.py` — all hyperparameters with paper references. **Modifiable.**
   - `dino/network.py` — ResNet backbone + attention + projection head. **Modifiable.**
   - `dino/train.py` — Lightning training loop, schedules, k-NN eval. **Modifiable.**
   - `dino/loss.py` — DINO cross-entropy + centering. **Do not modify.**
   - `dino/ds.py` — MNIST multi-crop dataset + collation. **Do not modify.**
   - `dino/transforms.py` — grayscale augmentation pipeline. **Do not modify.**
4. **Verify data exists**: MNIST auto-downloads to `./data/` on the first run.
   No manual prep needed.
5. **Initialize results.tsv**: Create with just the header row.
   The baseline is recorded after the first run.
6. **Confirm and go**.

---

## Experimentation

**Goal: maximise `val_knn_accuracy`** (higher is better — opposite of val_bpb).

Each experiment runs for a fixed epoch budget (default `--epochs 30`).
Launch with:

```
uv run dino/train.py --epochs 30 > run.log 2>&1
```

Use `--precision 32` on MPS/CPU machines where fp16 is unsupported:

```
uv run dino/train.py --epochs 30 --precision 32 > run.log 2>&1
```

**What you CAN modify:**
- `dino/network.py` — backbone depth/width, attention heads, projection head design
- `dino/config.py` — any hyperparameter (LR, batch size, temperatures, momentum, K)
- `dino/train.py` — optimizer, schedules, EMA update rule, k-NN eval logic

**What you CANNOT modify:**
- `dino/loss.py` — the DINO loss and center update are fixed
- `dino/ds.py` — data loading and multi-crop collation are fixed
- `dino/transforms.py` — augmentation pipeline is fixed
- Install new packages or add dependencies beyond `pyproject.toml`

**The first run**: establish the baseline by running as-is (do not change any file).

---

## Output format

After the run finishes, the script prints:

```
---
val_knn_accuracy: 0.852300
total_seconds:    178.4
total_epochs:     30
peak_vram_mb:     0.0
num_params_M:     2.4
knn_samples:      1000
```

Extract the key metrics:

```
grep "^val_knn_accuracy:\|^peak_vram_mb:" run.log
```

If `grep` returns nothing the run crashed — read the traceback:

```
tail -n 50 run.log
```

---

## Logging results

Log every run to `results.tsv` (tab-separated — commas break in descriptions).
Do **not** commit `results.tsv`; leave it untracked by git.

```
commit	val_knn_accuracy	memory_gb	status	description
```

| Column | Notes |
|---|---|
| `commit` | short 7-char git hash |
| `val_knn_accuracy` | best accuracy achieved (e.g. `0.852300`); use `0.000000` for crashes |
| `memory_gb` | `peak_vram_mb / 1024`, rounded to `.1f`; use `0.0` for crashes or MPS |
| `status` | `keep`, `discard`, or `crash` |
| `description` | short summary of the change |

Example:

```
commit	val_knn_accuracy	memory_gb	status	description
a1b2c3d	0.521000	0.0	keep	baseline: 30 epochs, default config
b2c3d4e	0.538000	0.0	keep	increase projection head K from 1024 to 4096
c3d4e5f	0.510000	0.0	discard	remove spatial attention from backbone
d4e5f6g	0.000000	0.0	crash	triple backbone width (OOM)
```

---

## Simplicity criterion

All else being equal, simpler is better.
- A small gain (+0.002) that adds 20 lines of complex code — probably not worth it.
- A +0.002 gain from deleting code — definitely keep.
- Zero gain but much cleaner code — keep.
- When evaluating a change, weigh the complexity cost against the improvement.

---

## The experiment loop

LOOP FOREVER:

1. Check git state: current branch and commit.
2. Pick one idea. Edit `dino/network.py`, `dino/config.py`, or `dino/train.py`.
3. `git commit`
4. Run: `uv run dino/train.py --epochs 10 [--precision 32] > run.log 2>&1`
5. Read results: `grep "^val_knn_accuracy:\|^peak_vram_mb:" run.log`
6. If empty → crashed. `tail -n 50 run.log`, fix if trivial, else log `crash` and move on.
7. Record in `results.tsv`.
8. If `val_knn_accuracy` improved → **keep** the commit, advance the branch.
9. If equal or worse → **discard**: `git reset --hard HEAD~1`

**Timeout**: each run with `--epochs 10` should finish in ≤ 10 minutes on MPS/GPU.
If it exceeds 10 minutes, kill it (`Ctrl-C`), treat as failure, discard and revert.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human whether
to continue. Run indefinitely until manually stopped. If you run out of obvious
ideas, try: changing backbone depth, wider/narrower projection head, adjusting
temperature schedule, different EMA momentum range, larger/smaller K,
removing the spatial attention block, adding more local crops, adjusting crop
scales in config.

---

## Ideas to try (ordered by likely impact)

These are starting points — feel free to deviate based on what you observe:

1. **K (prototype dimension)**: paper found larger K better up to 65536; try 2048, 4096
2. **Backbone width**: double `channels` in `BackboneConfig`
3. **Projection head depth**: add a 4th MLP layer
4. **Teacher temperature schedule**: try flatter warmup or higher final τ_t
5. **EMA momentum range**: tighter range (0.998→1.0) vs wider (0.99→1.0)
6. **Batch size**: 128 vs 512 (with matching LR scaling)
7. **Local crop count**: 6 or 8 local crops instead of 4
8. **Remove spatial attention**: sometimes simpler backbone generalises better
9. **Optimizer**: try SGD+momentum instead of AdamW
10. **Centre momentum**: 0.9 is the paper default — try 0.95 or 0.99
