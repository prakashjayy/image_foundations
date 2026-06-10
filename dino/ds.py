"""
MNIST multi-crop dataset for DINO.

Each __getitem__ returns (crops, label) where crops is a flat list:
  [global_0, global_1, local_0, ..., local_{n-1}]
  Every element is a (1, H, W) float tensor, normalised.

The DataLoader collate function re-organises this into:
  crops : list of V tensors, each (B, 1, H, W)   — one tensor per view
  labels: (B,) int64 tensor

V = n_global_crops + n_local_crops  (6 by default: 2 global + 4 local)
B = batch size

Only global crops are passed to the teacher; local + global go to the student.
See loss.py for how these are consumed.
"""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from dino.config import DINOConfig, cfg as default_cfg
from dino.transforms import global_transform, local_transform, MNIST_MEAN, MNIST_STD


class MNISTMultiCrop(Dataset):
    """Wraps torchvision MNIST and applies n_global + n_local random crop transforms."""

    def __init__(self, cfg: DINOConfig, train: bool = True, root: str = "./data"):
        self.base = MNIST(root=root, train=train, download=True)
        # Each call to the same transform object samples a fresh random crop,
        # so a single transform instance suffices per view type.
        self.global_tfms = [global_transform(cfg.data) for _ in range(cfg.data.n_global_crops)]
        self.local_tfms  = [local_transform(cfg.data)  for _ in range(cfg.data.n_local_crops)]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]          # img: PIL Image mode 'L', 28×28
        crops = (
            [t(img) for t in self.global_tfms]   # 2 × (1, 28, 28)
            + [t(img) for t in self.local_tfms]  # 4 × (1, 14, 14)
        )
        return crops, label


def collate_multicrop(batch: list) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Transpose (samples × views) into (views × samples).

    DataLoader default collate would try to stack (B, V, ...) but crops have
    different spatial sizes (global vs local), so we collate per-view instead.

    Returns:
        crops : list of V tensors each (B, 1, H, W)
        labels: (B,) int64
    """
    n_views = len(batch[0][0])
    crops  = [torch.stack([item[0][v] for item in batch]) for v in range(n_views)]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return crops, labels


def build_dataloader(
    cfg: DINOConfig,
    train: bool = True,
    root: str = "./data",
    num_workers: int | None = None,
) -> DataLoader:
    dataset = MNISTMultiCrop(cfg, train=train, root=root)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        num_workers=cfg.train.num_workers if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=collate_multicrop,
        drop_last=train,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    N_SHOW    = 8
    PAD       = 2       # pixels between cells in make_grid
    DATA_ROOT = "./data"

    loader = build_dataloader(default_cfg, train=True, root=DATA_ROOT, num_workers=0)
    crops, labels = next(iter(loader))
    # crops: list[n_views] of (B, 1, H, W) tensors

    n_global    = default_cfg.data.n_global_crops   # 2
    n_views     = len(crops)                        # 6
    global_size = default_cfg.data.global_crop_size # 28

    def denorm(t: torch.Tensor) -> torch.Tensor:
        return (t * MNIST_STD[0] + MNIST_MEAN[0]).clamp(0.0, 1.0)

    def pad_to_global(t: torch.Tensor) -> torch.Tensor:
        """Centre a (1, H, W) local crop on a black (1, global_size, global_size) canvas."""
        _, h, w = t.shape
        canvas = torch.zeros(1, global_size, global_size)
        y0 = (global_size - h) // 2
        x0 = (global_size - w) // 2
        canvas[:, y0:y0 + h, x0:x0 + w] = t
        return canvas

    # Build flat list: [row0_col0, row0_col1, ..., row0_col5, row1_col0, ...]
    # Each entry is a (1, global_size, global_size) tensor, denormalised.
    tiles = []
    for row in range(N_SHOW):
        for col in range(n_views):
            img = denorm(crops[col][row])           # (1, H, W)
            if col >= n_global:
                img = pad_to_global(img)            # local: centre on black canvas
            tiles.append(img)

    # make_grid arranges tiles in rows of n_views, with PAD black pixels between cells.
    # Output shape: (1, H_grid, W_grid)
    grid = make_grid(torch.stack(tiles), nrow=n_views, padding=PAD, pad_value=0.0)
    # make_grid always returns (3, H, W) — duplicate channels for grayscale input.
    # Take one channel since all three are equal.
    grid_np = grid[0].numpy()           # (H_grid, W_grid)

    # Draw a white divider in the padding gap between the last global and first local col.
    # make_grid layout: left_pad | col0 | pad | col1 | pad | col2 | ...
    # Column c starts at pixel: PAD + c * (global_size + PAD)
    # Gap before col n_global runs from: PAD + (n_global-1)*(global_size+PAD) + global_size
    gap_x = PAD + (n_global - 1) * (global_size + PAD) + global_size   # = 60 for our config
    grid_np[:, gap_x : gap_x + PAD] = 0.55   # mid-grey divider in the existing gap

    # ── Plot ───────────────────────────────────────────────────────────────
    h_px, w_px = grid_np.shape
    # Each view is ~1.1 inches wide; extra 0.4 inches on the left for row labels.
    fig, ax = plt.subplots(figsize=(n_views * 1.1 + 0.4, N_SHOW * 1.1 + 0.5))
    fig.patch.set_facecolor("black")
    ax.imshow(grid_np, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal",
              interpolation="nearest")
    ax.set_axis_off()

    # Column labels: position in data-pixel space (imshow sets 0,0 at top-left).
    col_labels = (
        [f"Global {i}" for i in range(n_global)]
        + [f"Local {i}" for i in range(n_views - n_global)]
    )
    for col, label in enumerate(col_labels):
        x_center = PAD + col * (global_size + PAD) + global_size / 2
        color = "#aaddff" if col < n_global else "#ffddaa"
        ax.text(x_center, -3, label, color=color,
                fontsize=8, ha="center", va="bottom",
                transform=ax.transData, clip_on=False)

    # Row labels (digit class) to the left of the first column.
    for row in range(N_SHOW):
        y_center = PAD + row * (global_size + PAD) + global_size / 2
        ax.text(-3, y_center, str(labels[row].item()),
                color="white", fontsize=8, ha="right", va="center",
                transform=ax.transData, clip_on=False)

    out_path = os.path.join(os.path.dirname(__file__), "crops_preview.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="black")
    print(f"Saved {out_path}")
    plt.show()
