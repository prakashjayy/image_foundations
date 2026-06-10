"""
Augmentation transforms for MNIST DINO training.

Paper uses BYOL augmentations (Sec 3 Impl.): color jitter, Gaussian blur,
solarization, random flip.  For single-channel grayscale we adapt:

  Color jitter    → RandomAutocontrast + random brightness/contrast
                    (ColorJitter works on grayscale but only affects brightness
                    and contrast; we use the grayscale-native equivalents)
  Solarization    → skipped (pixel-inversion on digits destroys structure)
  Horizontal flip → skipped; MNIST digits have fixed orientation
                    (6 flipped = 9, etc.) — flipping creates out-of-distribution
                    views that hurt representation quality on this dataset
  Gaussian blur   → kept; paper applies it to global views with p=0.5

Two scales of crops (paper Appendix E):
  Global (28×28, scale 0.5–1.0): strong blur + contrast; reliable teacher targets
  Local  (14×14, scale 0.2–0.5): lighter aug; student must match teacher from
                                  small, partial views (local-to-global learning)
"""

from torchvision import transforms

from dino.config import DataConfig

# MNIST pixel statistics (computed over the full training set).
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def global_transform(cfg: DataConfig) -> transforms.Compose:
    """Strong augmentation for the 2 global views fed to teacher + student.

    Paper: "global views use a larger resolution and stronger augmentations
    to provide reliable teacher targets" (Sec 3.1, Appendix E).
    Scale range (0.5, 1.0) chosen for MNIST (paper uses 0.32–1.0 at 224px).
    """
    return transforms.Compose([
        # Primary spatial augmentation — same mechanism as paper but smaller
        # scale range since MNIST digits are already small.
        transforms.RandomResizedCrop(
            cfg.global_crop_size,
            scale=cfg.global_crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        # Grayscale analog of ColorJitter. Paper uses p=0.8 for color distortion;
        # brightness+contrast adjustment gives equivalent view diversity for digits.
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8
        ),
        # Gaussian blur: paper applies with p=1.0 on first global view,
        # p=0.1 on second (BYOL schedule). We use p=0.5 as a single shared
        # probability since both global transforms are identical here.
        # Kernel 3 is the smallest odd value meaningful for 28×28 images.
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.5
        ),
        # Random erasing: replaces a rectangle with the mean value, forcing
        # the network to look at context rather than one dominant stroke.
        # Not in paper, but standard for small-image self-supervised setups.
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value=0.0),
    ])


def local_transform(cfg: DataConfig) -> transforms.Compose:
    """Lighter augmentation for the n_local small views fed only to student.

    Paper: local crops are small (96px vs 224px global) and use the same
    augmentation pipeline as global but at a smaller scale (0.05–0.32).
    For MNIST we reduce blur probability and skip contrast distortion
    so the local patches still contain recognisable digit structure.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.local_crop_size,
            scale=cfg.local_crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        # Light contrast variation only — local crops are already small so
        # heavy distortion removes all structure.
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
        ),
        # Blur at lower probability: local patches are 14×14, strong blur
        # would collapse distinct digit strokes into uniform grey.
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
