"""
DINO configuration for MNIST training.

Paper: "Emerging Properties in Self-Supervised Vision Transformers"
       Caron et al., 2021 (arXiv:2104.14294)

Every parameter below notes the paper value and the rationale for adapting it to MNIST
(28×28 grayscale, 10 classes, ~60K training images on a single GPU).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Paper trains on ImageNet (224×224 RGB). MNIST is 28×28 grayscale.
    # We resize to 32×32 so the network has cleaner power-of-2 spatial dims.
    image_size: int = 32

    # Single channel (grayscale). Paper uses 3-channel RGB.
    in_channels: int = 1

    # Multi-crop: paper uses 2 global views (224×224) + 6 local views (96×96).
    # Scale for global views: (0.32, 1.0). Paper found 0.32 optimal (Appendix E).
    # For 32×32 MNIST: global crops sampled at (0.5, 1.0) → 16–32px range.
    global_crop_size: int = 28
    global_crop_scale: tuple = (0.5, 1.0)
    n_global_crops: int = 2          # Paper: always 2 global views fed to teacher + student

    # Local views: paper (0.05, 0.32) at 96×96. For MNIST: (0.2, 0.5) → 6–14px.
    local_crop_size: int = 14
    local_crop_scale: tuple = (0.2, 0.5)
    n_local_crops: int = 6           # Paper uses 6–10; try 6 for more view diversity

    # k for k-NN evaluation. Paper uses k=20 (Appendix F.1, "consistently best").
    knn_k: int = 20

    # Temperature for k-NN weighted voting (Appendix F.1, same as Wu et al. 2018).
    knn_temperature: float = 0.07


# ---------------------------------------------------------------------------
# Backbone (ResNet with self-attention)
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    # Paper backbone options: ResNet-50 (2048-dim) or ViT-S/16 (384-dim).
    # For MNIST we use a lightweight ResNet (≈ResNet-18 depth) with
    # a self-attention block inserted before the final pooling, giving
    # the model spatial awareness similar to the CLS token in ViTs.

    # Feature dimension after global average pooling.
    # Paper: 2048 for ResNet-50, 384 for ViT-S. Small MNIST network → 256.
    feat_dim: int = 512

    # ResNet layer widths [layer1, layer2, layer3, layer4].
    # Kept small: MNIST has far less visual complexity than ImageNet.
    channels: tuple = (64, 128, 256, 512)

    # Number of residual blocks per stage.
    blocks: tuple = (1, 1, 2, 1)

    # Self-attention is placed after layer4 (before pooling).
    # Paper (ViT): 12 attention blocks, 6 heads for ViT-S.
    # We use a single multi-head self-attention layer; 8 heads matches
    # feat_dim=512 cleanly (64 dims/head).
    attn_heads: int = 8

    # Dropout in the attention module (not used in paper, added for small dataset).
    attn_dropout: float = 0.0


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

@dataclass
class HeadConfig:
    # Paper (Sec 3 / Appendix C): 3-layer MLP (hidden=2048, GELU) →
    #   L2 norm → weight-normalized FC with K output dims.
    # The L2 bottleneck is critical: training collapses without it when depth ≥ 3
    # (Appendix C ablation).

    # Hidden dimension of the MLP layers.
    # Paper: 2048. Scaled down proportionally to our smaller backbone (256-dim).
    hidden_dim: int = 512

    # Number of MLP layers before the L2 bottleneck.
    # Paper: 3 (total 4 linear layers including the final weight-norm layer).
    n_layers: int = 2

    # Bottleneck dimension (output of L2 norm, input of weight-norm FC).
    # Paper: 256. Kept the same here; not expensive regardless of K.
    bottleneck_dim: int = 128

    # Output dimensionality K (the "number of prototypes").
    # Paper: K=65536 for ImageNet (ablation: larger K generally better, Appendix C).
    # MNIST has 10 classes and 60K images; K=1024 gives ≥100 prototypes per class
    # on average — sufficient to learn a rich representation.
    out_dim: int = 1024

    # Activation in MLP. Paper uses GELU throughout (consistent with ViT default).
    activation: str = "gelu"

    # No batch normalization. Paper (Appendix C): "BN-free system" works better
    # with ViT and has little impact on ResNet variants — we follow this.
    use_bn: bool = False


# ---------------------------------------------------------------------------
# Temperatures
# ---------------------------------------------------------------------------

@dataclass
class TemperatureConfig:
    # Student softmax temperature τ_s (Eq. 1 in paper).
    # Paper: τ_s = 0.1 (sharper student distribution → stronger gradient signal).
    student_temp: float = 0.1

    # Teacher softmax temperature τ_t — controls sharpness of targets.
    # Paper: linear warmup from 0.04 → 0.07 over first 30 epochs (Sec 3 /
    # Appendix D). Starting low prevents collapse in early training.
    # For MNIST 100-epoch run, warmup over first 10 epochs.
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: int = 10   # Paper: 30 epochs for 300-epoch run


# ---------------------------------------------------------------------------
# EMA / Momentum Teacher
# ---------------------------------------------------------------------------

@dataclass
class MomentumConfig:
    # Teacher EMA update rule: θ_t ← λ·θ_t + (1−λ)·θ_s
    # Paper (Sec 3): λ follows a cosine schedule from 0.996 → 1.0.
    # This "Polyak-Ruppert averaging" causes the teacher to consistently
    # outperform the student throughout training (Fig. 6, Appendix D).
    # Momentum=0 → immediate collapse; this is the single most critical component
    # (Table 7, row 2: k-NN = 0.1% without momentum).
    momentum_start: float = 0.996
    momentum_end: float = 1.0         # Reaches 1 at end of training (frozen teacher)

    # Center EMA momentum m in: C ← m·C + (1−m)·mean(teacher_outputs).
    # Paper / Appendix D ablation: m=0.9 is optimal; m=0.999 collapses.
    # Centering prevents one dimension from dominating (Sec 3.2).
    center_momentum: float = 0.9


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    # Paper: AdamW optimizer (Sec 3 Implementation Details).
    name: str = "adamw"

    # Base learning rate. Paper uses lr = 0.0005 × batch_size / 256.
    # With batch_size=256 → lr = 0.0005. Linear scaling rule [Goyal et al. 2017].
    base_lr: float = 5e-4

    # AdamW betas — paper does not specify, standard defaults assumed.
    betas: tuple = (0.9, 0.999)

    # Weight decay: cosine schedule from 0.04 → 0.4 (Sec 3 Implementation).
    # Controls regularisation progressively through training.
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

@dataclass
class ScheduleConfig:
    # Warmup: paper linearly ramps LR for the first 10 epochs (Sec 3 Impl.).
    # For 100-epoch MNIST run we use the same 10-epoch warmup.
    warmup_epochs: int = 2

    # After warmup: cosine decay to 0 (Sec 3 Impl.).
    # Paper trains 300–800 epochs on ImageNet. MNIST converges faster;
    # 100 epochs is sufficient (analogous to the 100-epoch ablation, Table 8).
    total_epochs: int = 100

    # Minimum LR at end of cosine decay. Paper implies decay to ~0.
    min_lr: float = 1e-6


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paper: batch_size=1024 over 16 GPUs for ViT-S/16.
    # Single-GPU MNIST training; 256 is large enough relative to the dataset.
    # Paper (Table 9): bs=128 gives -2% k-NN vs bs=1024; 256 is a good trade-off.
    batch_size: int = 256

    # Paper uses float32 / mixed precision not explicitly stated; we enable
    # fp16 via Lightning for training speed.
    precision: str = "16-mixed"

    # Number of DataLoader workers.
    num_workers: int = 4

    # Gradient clipping. Paper does not mention it, but useful for stability
    # with small datasets.
    gradient_clip_val: float = 3.0

    # Logging interval (every N steps).
    log_every_n_steps: int = 50


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Run k-NN evaluation every N epochs during training.
    knn_eval_every_n_epochs: int = 1

    # Fraction of training data to use as the k-NN feature bank.
    # For MNIST the full training set (60K) is small enough to use entirely.
    knn_feature_bank_fraction: float = 1.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class DINOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Effective LR after linear scaling: lr = base_lr * batch_size / 256
    @property
    def lr(self) -> float:
        return self.optimizer.base_lr * self.train.batch_size / 256


# Default config instance ready to import
cfg = DINOConfig()
