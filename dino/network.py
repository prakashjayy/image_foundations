"""
DINO network for MNIST.

Architecture: g = h ∘ f  (paper Sec 3)
  f : ResNetBackbone  — lightweight ResNet with a self-attention block
  h : DINOHead        — MLP projection head with L2 bottleneck + weight-norm FC
  g : DINONet         — combines f and h; used identically for student and teacher
"""

import os
import sys

# Allow `uv run dino/network.py` — adds project root to sys.path before dino imports.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from dino.config import BackboneConfig, DINOConfig, HeadConfig


# ---------------------------------------------------------------------------
# Backbone building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Standard pre-activation residual block (He et al. 2016).

    Paper uses ResNet-50 as the convnet backbone (Sec 3).  We use BasicBlock
    (two 3×3 convs) instead of Bottleneck — appropriate for a small network.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut when spatial size or channel count changes.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SpatialAttention(nn.Module):
    """Multi-head self-attention over flattened spatial tokens.

    Paper uses ViT which is entirely attention-based (Table 1, 12 blocks).
    Here we add one MHSA block after the last ResNet stage — the spatial
    feature map (4×4 = 16 tokens for global crops, 2×2 = 4 for local crops)
    is treated as a token sequence, mirroring how ViT patch tokens attend
    to each other before the [CLS] readout.

    LayerNorm + residual connection follows standard Transformer convention
    (paper Sec 3.2, "pre-norm" variant from ViT).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims into a token sequence: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended          # residual connection
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Backbone  f
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """Lightweight ResNet followed by a single self-attention block.

    Spatial trace for a 32×32 input (global crop):
        stem           →  32×32
        layer1 stride1 →  32×32
        layer2 stride2 →  16×16
        layer3 stride2 →   8×8
        layer4 stride2 →   4×4   ← attention applied here (16 tokens)
        AdaptiveAvgPool→   1×1   → feat_dim-d vector

    For 14×14 local crops the same path gives a 2×2 attention grid (4 tokens),
    which still works because AdaptiveAvgPool handles arbitrary spatial size.
    """

    def __init__(self, cfg: BackboneConfig, in_channels: int = 1):
        super().__init__()
        ch = cfg.channels   # (32, 64, 128, 256)
        bl = cfg.blocks     # (1,  1,  2,   1)

        # Stem: single 3×3 conv with no striding — MNIST is already 32×32, a
        # large-kernel / strided stem like ImageNet ResNet would over-shrink it.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(ch[0], ch[0], bl[0], stride=1)
        self.layer2 = self._make_layer(ch[0], ch[1], bl[1], stride=2)
        self.layer3 = self._make_layer(ch[1], ch[2], bl[2], stride=2)
        self.layer4 = self._make_layer(ch[2], ch[3], bl[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)     # (B, feat_dim)
        return x


# ---------------------------------------------------------------------------
# Projection head  h
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """DINO projection head: n-layer MLP → L2 norm → weight-norm FC.

    Paper (Sec 3 / Appendix C):
    - 3-layer MLP, hidden=2048, GELU activations, no BN.
    - "Last layer of the MLP is without GELU."
    - ℓ2 normalisation between MLP and final FC (the "L2 bottleneck").
      Ablation (Appendix C): training collapses without it at depth ≥ 3.
    - Final layer uses weight normalisation [Salimans & Kingma 2016];
      weight_g is frozen so only the direction is learnt.
    - No bias on the final layer.
    """

    def __init__(self, cfg: HeadConfig, in_dim: int):
        super().__init__()
        act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU(inplace=True)

        # Build the MLP: (n_layers - 1) hidden layers + one bottleneck layer.
        layers: list[nn.Module] = []
        cur = in_dim
        for i in range(cfg.n_layers):
            is_last = i == cfg.n_layers - 1
            out = cfg.bottleneck_dim if is_last else cfg.hidden_dim
            layers.append(nn.Linear(cur, out))
            if not is_last:
                layers.append(act)
            cur = out
        self.mlp = nn.Sequential(*layers)

        # Weight-normalised output layer (no bias).
        # Paper: "weight normalised fully connected layer with K dimensions."
        last = nn.Linear(cfg.bottleneck_dim, cfg.out_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(last)
        # Freeze weight_g (magnitude) — only train direction.
        # Matches norm_last_layer=True in the official DINO code.
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)    # L2 bottleneck (paper Appendix C)
        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# Full network  g = h ∘ f
# ---------------------------------------------------------------------------

class DINONet(nn.Module):
    """Student / teacher network: backbone + projection head.

    Both student and teacher share this exact class with the same architecture.
    They differ only in how their parameters are updated:
      - Student : standard gradient updates.
      - Teacher : EMA of the student (no gradients, updated in train.py).

    Paper (Sec 3): "we do not have a predictor, resulting in the exact same
    architecture in both student and teacher networks."
    """

    def __init__(self, cfg: DINOConfig):
        super().__init__()
        self.backbone = ResNetBackbone(cfg.backbone, in_channels=cfg.data.in_channels)
        self.head = DINOHead(cfg.head, in_dim=cfg.backbone.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw (pre-softmax) logits of shape (B, K)."""
        return self.head(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only features used for k-NN evaluation (paper Appendix F.1)."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Helper: build student + teacher pair
# ---------------------------------------------------------------------------

def build_student_teacher(cfg: DINOConfig) -> tuple[DINONet, DINONet]:
    """Create student and teacher with identical initialisation.

    Paper (Algorithm 1): "gt.params = gs.params" at the start of training.
    Teacher parameters are frozen; EMA update happens in the training loop.

    We build the teacher as a separate DINONet and copy weights via state_dict
    rather than deepcopy — deepcopy fails after weight_norm hooks are installed
    (PyTorch known issue; deepcopy of non-leaf tensors is not supported).
    """
    student = DINONet(cfg)
    teacher = DINONet(cfg)
    teacher.load_state_dict(student.state_dict())

    # Teacher receives no gradients — updated only via EMA.
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dino.config import cfg
    from dino.loss import DINOLoss

    torch.manual_seed(0)
    student, teacher = build_student_teacher(cfg)
    student.eval()
    teacher.eval()

    B = 4
    # Paper: 2 global crops (resized to global_crop_size) + n_local local crops.
    global_crops = [torch.randn(B, cfg.data.in_channels, cfg.data.global_crop_size,
                                cfg.data.global_crop_size) for _ in range(cfg.data.n_global_crops)]
    local_crops  = [torch.randn(B, cfg.data.in_channels, cfg.data.local_crop_size,
                                cfg.data.local_crop_size) for _ in range(cfg.data.n_local_crops)]

    with torch.no_grad():
        student_out = [student(v) for v in global_crops + local_crops]
        teacher_out = [teacher(v) for v in global_crops]

    print("=" * 55)
    print("Network output shapes")
    print("=" * 55)
    print(f"  Student  ({cfg.data.n_global_crops} global + {cfg.data.n_local_crops} local crops)")
    for i, s in enumerate(student_out):
        tag = "global" if i < cfg.data.n_global_crops else "local"
        print(f"    [{tag} {i}]  {tuple(s.shape)}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
    print(f"  Teacher  ({cfg.data.n_global_crops} global crops only)")
    for i, t in enumerate(teacher_out):
        print(f"    [global {i}]  {tuple(t.shape)}  "
              f"min={t.min():.3f}  max={t.max():.3f}")

    print()
    print("Backbone features (used for k-NN eval)")
    feats = student.get_features(global_crops[0])
    print(f"    {tuple(feats.shape)}  — backbone feat_dim={cfg.backbone.feat_dim}")

    print()
    print("Parameter counts")
    def n_params(m): return sum(p.numel() for p in m.parameters())
    print(f"    backbone : {n_params(student.backbone):>9,}")
    print(f"    head     : {n_params(student.head):>9,}")
    print(f"    total    : {n_params(student):>9,}")
    print(f"    teacher frozen: "
          f"{all(not p.requires_grad for p in teacher.parameters())}")

    print()
    print("DINO loss (one batch, teacher_temp=0.04)")
    # Re-run with grad for loss
    student.train()
    student_out_g = [student(v) for v in global_crops + local_crops]
    teacher_out_g = [teacher(v) for v in global_crops]
    loss_fn = DINOLoss(cfg)
    loss = loss_fn(student_out_g, teacher_out_g, teacher_temp=0.04)
    print(f"    loss = {loss.item():.4f}  (≈ ln({cfg.head.out_dim}) = {torch.log(torch.tensor(float(cfg.head.out_dim))):.4f} at init)")
    print("=" * 55)
