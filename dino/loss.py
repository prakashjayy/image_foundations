"""
DINO loss: cross-entropy between sharpened teacher and student outputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA FLOW  (one training step, batch of B images)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  B images in batch
  │
  └─ augment each image into 6 crop views (ds.py)
       ├─ global view 0  (28×28)  ──┐
       ├─ global view 1  (28×28)  ──┼──► Teacher ──► teacher_out  list of 2 tensors
       ├─ local  view 2  (14×14)  ──┤                each shape (B, K)
       ├─ local  view 3  (14×14)  ──┤
       ├─ local  view 4  (14×14)  ──┤
       └─ local  view 5  (14×14)  ──┴──► Student ──► student_out  list of 6 tensors
                                                      each shape (B, K)

  B = batch size (e.g. 256).  Each row in a (B, K) tensor is one image.
  K = projection head output dim (e.g. 1024 prototypes).
  Student and teacher share the same architecture → same output shape (B, K).

  Why does teacher see only global crops?
    Local crops are small, partial views — too noisy to be reliable targets.
    The teacher provides high-quality global targets; the student is forced to
    predict them even from low-resolution local patches.
    Paper Sec 3.1: "only the global views are passed through the teacher."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSS-VIEW LOSS PAIRS  (t_idx ≠ s_idx)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  teacher[0] paired with student[1, 2, 3, 4, 5]  →  5 pairs
  teacher[1] paired with student[0, 2, 3, 4, 5]  →  5 pairs
                                                    ─────────
                                                    10 pairs total

  Same-view pairs (t=0,s=0) and (t=1,s=1) are skipped — a network
  that just copies its input would trivially minimise those.

  The "local-to-global" pairs (e.g. teacher[0] × student[2..5]) are the
  key signal: student must predict global semantics from a small crop,
  forcing scale-invariant representations.
  Paper Table 8: multi-crop adds +2 % k-NN accuracy at the same compute.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PER-PAIR COMPUTATION  H(t, s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  teacher_prob    = softmax( (teacher_logit − C) / τ_t )   shape (B, K)
  student_logprob = log_softmax( student_logit  / τ_s )    shape (B, K)
  H               = − mean_over_B( sum_over_K( teacher_prob · student_logprob ) )

  τ_t = 0.04 → 0.07 warmed up over first 10 epochs  (paper: 30 epochs on ImageNet)
  τ_s = 0.1  fixed                                   (paper Sec 3, Implementation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLAPSE PREVENTION  (Sec 3.2 / Fig. 7 / Appendix D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Two opposing forces that must both be present:

  Centering  — subtract running mean C (shape K) from teacher logits.
               Stops one prototype dimension dominating all outputs.
               Alone → pushes distribution toward uniform (high entropy).

  Sharpening — low τ_t concentrates probability mass on a few prototypes.
               Alone → one dimension collapses to probability 1.

  Center update after every batch (paper Eq. 4):
    C ← m · C + (1 − m) · mean_over_batch(teacher_logits)
    m = 0.9  (Appendix D: m=0.999 collapses, m=0.9 is optimal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dino.config import DINOConfig


class DINOLoss(nn.Module):
    """See module docstring for full data-flow and design rationale."""

    def __init__(self, cfg: DINOConfig):
        super().__init__()
        self.student_temp    = cfg.temperature.student_temp
        self.center_momentum = cfg.momentum.center_momentum

        # Non-learnable buffer; persists across batches and survives checkpointing.
        self.register_buffer("center", torch.zeros(1, cfg.head.out_dim))

    def forward(
        self,
        student_out: list[torch.Tensor],
        teacher_out: list[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        # Teacher: center then sharpen (stop-gradient already applied upstream).
        teacher_probs = [
            F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
            for t in teacher_out
        ]
        # Student: sharpen only.
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_out
        ]

        # Average cross-entropy over all cross-view (teacher, student) pairs.
        # Same-view pairs are skipped (paper Algorithm 1).
        loss = torch.tensor(0.0, device=student_out[0].device)
        n_pairs = 0
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                if s_idx == t_idx:
                    continue
                loss += -(t_prob * s_log_prob).sum(dim=-1).mean()
                n_pairs += 1
        loss = loss / n_pairs

        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: list[torch.Tensor]) -> None:
        batch_mean = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center = (
            self.center_momentum * self.center
            + (1 - self.center_momentum) * batch_mean
        )
