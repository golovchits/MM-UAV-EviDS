#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# EviDS-UAV: Two-Stream YOLOX with DS Fusion at Decision Level
#
# Extends YOLOX2: heads produce Dirichlet parameters; DS fusion combines
# them at the decision level. Optional temporal gating for efficiency.
#
# Training: each head learns independently via EDL loss (standard two-stream).
# Inference: per-anchor DS fusion of RGB + IR Dirichlet outputs into a single
#            fused detection tensor.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_head_evidential import YOLOXHeadEvidential, softplus_evidence
from .yolo_pafpn2 import YOLOPAFPN2


class YOLOX2DS(nn.Module):
    """Two-stream YOLOX with DS evidence fusion at decision level.

    Compared to YOLOX2:
      - Heads are evidential (YOLOXHeadEvidential)
      - DS fusion combines per-modality Dirichlet outputs into fused belief
      - Optional temporal gating for efficiency
      - Training: per-modality EDL loss (same as YOLOX2)
      - Inference: returns DS-fused detection tensor
    """

    def __init__(self, backbone=None, head=None, head2=None,
                 ds_fusion=None, temporal_gate=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN2()
        if head is None:
            head = YOLOXHeadEvidential(1)
        if head2 is None:
            head2 = YOLOXHeadEvidential(1)

        self.backbone = backbone
        self.head = head
        self.head2 = head2
        self.ds_fusion = ds_fusion
        self.temporal_gate = temporal_gate

    @property
    def num_classes(self):
        return self.head.num_classes

    def forward(self, x1, x2, targets1=None, targets2=None):
        fpn_outs1, fpn_outs2 = self.backbone(x1, x2)

        if self.training:
            assert targets1 is not None and targets2 is not None

            loss1 = self.head(fpn_outs1, targets1, x1)
            loss2 = self.head2(fpn_outs2, targets2, x2)

            return {
                "total_loss": loss1[0],
                "iou_loss": loss1[1],
                "l1_loss": loss1[2],
                "conf_loss": loss1[3],
                "cls_loss": loss1[4],
                "num_fg": loss1[5],
                "total_loss2": loss2[0],
                "iou_loss2": loss2[1],
                "l1_loss2": loss2[2],
                "conf_loss2": loss2[3],
                "cls_loss2": loss2[4],
                "num_fg2": loss2[5],
            }

        # ── Inference ───────────────────────────────────────────────────
        outputs1 = self.head(fpn_outs1)      # [B, N, 5+K] decoded
        outputs2 = self.head2(fpn_outs2)     # [B, N, 5+K] decoded

        if self.ds_fusion is None:
            # No DS fusion at decision level (e.g. condition c: ADFM at feature level)
            return outputs1, outputs2

        # ── DS evidence fusion at decision level ────────────────────────
        cls_logits1 = self.head._last_cls_logits   # [B, N, K]
        cls_logits2 = self.head2._last_cls_logits  # [B, N, K]

        # Compute Dirichlet alphas with explicit background dimension
        alphas1 = self._logits_to_alphas(cls_logits1)  # [B, N, K+1]
        alphas2 = self._logits_to_alphas(cls_logits2)  # [B, N, K+1]

        b_fused, u_fused, C, fallback = self.ds_fusion(alphas1, alphas2)
        self._last_alphas1 = alphas1       # [B, N, K+1] — expose for diagnostics
        self._last_alphas2 = alphas2       # [B, N, K+1]
        self._last_conflict_C = C          # [B, N, 1] — expose for agree-vs-disagree analysis
        self._last_u_fused = u_fused       # [B, N, 1] — DS-fused uncertainty
        # b_fused: [B, N, K+1], drop bg belief (index -1) for detection output
        b_drone = b_fused[..., :self.num_classes]  # [B, N, K]

        # Build fused output: keep RGB reg/obj, replace cls with DS-fused belief
        fused_output = outputs1.clone()
        fused_output[..., 5:] = b_drone

        # Temporal gating (for condition f): suppress modality with high epistemic uncertainty
        if self.temporal_gate is not None:
            # Restrict uncertainty signal to foreground-like anchors (obj > 0.1).
            # Mean over all ~8400 anchors is degenerate: background anchors dominate
            # with u≈1.0, making the gate fire constantly regardless of content.
            u_rgb = self._mean_uncertainty(alphas1, outputs1[..., 4])
            u_ir = self._mean_uncertainty(alphas2, outputs2[..., 4])
            gate_rgb, gate_ir = self.temporal_gate(u_rgb, u_ir)
            self._gate_state = (gate_rgb, gate_ir)
            self._gate_u = (u_rgb, u_ir)

            if gate_rgb and not gate_ir:
                # RGB high-uncertainty: zero RGB evidence, redo DS with IR only.
                # Uniform prior alpha=1 = no evidence from RGB. DS reduces to IR belief.
                alphas1_uniform = torch.ones_like(alphas1)
                b_gated, _, _, _ = self.ds_fusion(alphas1_uniform, alphas2)
                fused_output[..., 5:] = b_gated[..., :self.num_classes]
            elif gate_ir and not gate_rgb:
                # IR high-uncertainty: zero IR evidence, redo DS with RGB only.
                alphas2_uniform = torch.ones_like(alphas2)
                b_gated, _, _, _ = self.ds_fusion(alphas1, alphas2_uniform)
                fused_output[..., 5:] = b_gated[..., :self.num_classes]

        # Return fused twice for evaluator compatibility (two dataloaders)
        return fused_output, fused_output

    @staticmethod
    def _logits_to_alphas(cls_logits):
        """Convert raw classification logits to full Dirichlet alphas [B, N, K+1].

        alpha_drone = softplus(logit) + 1
        alpha_bg    = 1  (uniform prior, zero evidence for background)
        """
        evidence = softplus_evidence(cls_logits)          # [B, N, K]
        alpha_drone = evidence + 1.0                      # [B, N, K]
        alpha_bg = torch.ones_like(alpha_drone[..., :1])  # [B, N, 1]
        return torch.cat([alpha_drone, alpha_bg], dim=-1) # [B, N, K+1]

    @staticmethod
    def _mean_uncertainty(alphas, obj_scores=None, obj_thresh=0.1):
        """Mean epistemic uncertainty for temporal gating.

        u = (K+1) / S  where S = sum(alpha_k)

        If obj_scores provided, restrict to anchors with objectness > obj_thresh.
        Falls back to full mean if no anchors pass the mask (empty frame).
        """
        S = alphas.sum(dim=-1)                   # [B, N]
        K_dim = alphas.shape[-1]                 # K+1
        u = K_dim / S.clamp(min=1e-8)            # [B, N]
        if obj_scores is not None:
            mask = obj_scores > obj_thresh        # [B, N]
            if mask.any():
                return u[mask].mean()
        return u.mean()                           # fallback: all anchors
