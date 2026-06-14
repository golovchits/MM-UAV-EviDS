#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# EviDS-UAV: Two-Stream YOLOX with Dirichlet Probability Averaging (condition d).
#
# Same OGAA-only FPN as condition (e). At inference, averages per-anchor
# Dirichlet class probabilities (p_k = alpha_k / S) from RGB and IR heads.
# This isolates the effect of the DS fusion rule: if (e) > (d), the gain is
# attributable to DS evidence combination rather than to having evidential heads.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_head_evidential import YOLOXHeadEvidential, softplus_evidence
from .yolo_pafpn2 import YOLOPAFPN2


class YOLOX2Average(nn.Module):
    """Two-stream YOLOX with Dirichlet probability averaging at decision level."""

    def __init__(self, backbone=None, head=None, head2=None):
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

        # ── Inference: average Dirichlet probabilities ──────────────────
        outputs1 = self.head(fpn_outs1)      # [B, N, 5+K] decoded
        outputs2 = self.head2(fpn_outs2)     # [B, N, 5+K] decoded

        cls_logits1 = self.head._last_cls_logits   # [B, N, K]
        cls_logits2 = self.head2._last_cls_logits  # [B, N, K]

        # Compute Dirichlet probabilities per modality
        evidence1 = softplus_evidence(cls_logits1)          # [B, N, K]
        alphas1 = evidence1 + 1.0
        S1 = alphas1.sum(dim=-1, keepdim=True) + 1.0       # +1 for implicit bg
        p1 = alphas1 / S1                                   # [B, N, K]

        evidence2 = softplus_evidence(cls_logits2)
        alphas2 = evidence2 + 1.0
        S2 = alphas2.sum(dim=-1, keepdim=True) + 1.0
        p2 = alphas2 / S2

        # Average Dirichlet probabilities (element-wise mean)
        p_avg = (p1 + p2) / 2.0  # [B, N, K]

        # Fused output: keep RGB reg/obj, replace cls with averaged probs
        fused_output = outputs1.clone()
        fused_output[..., 5:] = p_avg

        return fused_output, fused_output
