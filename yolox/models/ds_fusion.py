#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EviDS-UAV: Dempster-Shafer Evidence Fusion
#
# Converts per-modality Dirichlet parameters to belief masses and combines
# them using the Dempster-Shafer rule. Handles M >= 2 modalities via iterative
# combination. Falls back to most-certain single modality when conflict C > C_max.
#
# Alphas are [B, ..., K+1] where the last dimension is the explicit background
# alpha (alpha_bg = 1 by default). This makes DS conflict meaningful when K>1
# or when background evidence is learned.

import torch
import torch.nn as nn


def alpha_to_bpa(alphas):
    """Convert Dirichlet parameters to belief masses (Eq. 3 in methodology).

    Args:
        alphas: Dirichlet parameters, shape [..., M, K+1] or [..., K+1]
            where M = number of modalities, K+1 includes explicit background alpha.
            alpha_k = evidence_k + 1 (so alpha_bg = 1 when evidence_bg = 0).

    Returns:
        b: belief masses, same shape as alphas, b_k = (alpha_k - 1) / S
        u: uncertainty mass, shape [..., M, 1] or [..., 1]
            u = (K+1) / S  (dimensionality of the Dirichlet)
    """
    evidence = alphas - 1.0          # e_k = alpha_k - 1; e_bg = 0 when alpha_bg = 1
    S = alphas.sum(dim=-1, keepdim=True)  # total Dirichlet strength
    K_dim = alphas.shape[-1]         # K+1 (includes bg)
    b = evidence / S.clamp(min=1e-8)
    u = K_dim / S.clamp(min=1e-8)
    return b, u


def ds_combine(b1, u1, b2, u2, C_max=0.95):
    """Dempster-Shafer combination of two belief assignments (Eqs. 4-5).

    Args:
        b1, b2: belief masses, shape [..., K+1]
        u1, u2: uncertainty masses, shape [..., 1]

    Returns:
        b_fused: fused belief masses, shape [..., K+1]
        u_fused: fused uncertainty, shape [..., 1]
        C: conflict, shape [..., 1]
        fallback: bool, shape [..., 1]
    """
    # Conflict: C = sum_{i != j} b1_i * b2_j
    # Vectorised: C = sum(b1)*sum(b2) - sum(b1 * b2)
    b1_sum = b1.sum(dim=-1, keepdim=True)
    b2_sum = b2.sum(dim=-1, keepdim=True)
    C = b1_sum * b2_sum - (b1 * b2).sum(dim=-1, keepdim=True)
    C = C.clamp(0.0, 1.0)

    numerator = b1 * b2 + b1 * u2 + b2 * u1
    denom = 1.0 - C

    fallback = C > C_max
    safe_denom = denom.clamp(min=1e-8)
    b_fused = numerator / safe_denom
    u_fused = (u1 * u2) / safe_denom

    if fallback.any():
        b_fallback = torch.where(u1 <= u2, b1, b2)
        u_fallback = torch.where(u1 <= u2, u1, u2)
        b_fused = torch.where(fallback, b_fallback, b_fused)
        u_fused = torch.where(fallback, u_fallback, u_fused)

    return b_fused, u_fused, C, fallback


def fuse_modalities(alphas, C_max=0.95):
    """Iterative DS fusion for M >= 2 modalities.

    Args:
        alphas: Dirichlet parameters, shape [B, M, K+1]
        C_max: conflict threshold

    Returns:
        b_fused: fused belief masses, shape [B, K+1]
        u_fused: fused uncertainty, shape [B, 1]
        C_list: list of per-step conflict tensors
        fallback_mask: bool, shape [B, 1]
    """
    B, M, _ = alphas.shape
    b, u = alpha_to_bpa(alphas)  # [B, M, K+1], [B, M, 1]

    if M == 1:
        return b[:, 0], u[:, 0], [torch.zeros(B, 1)], torch.zeros(B, 1, dtype=torch.bool)

    b_fused = b[:, 0]
    u_fused = u[:, 0]
    C_list = []
    any_fallback = torch.zeros(B, 1, dtype=torch.bool, device=alphas.device)

    for m in range(1, M):
        b_fused, u_fused, C, fb = ds_combine(
            b_fused, u_fused, b[:, m], u[:, m], C_max
        )
        C_list.append(C)
        any_fallback = any_fallback | fb

    return b_fused, u_fused, C_list, any_fallback


class DSFusion(nn.Module):
    """Dempster-Shafer evidence fusion module.

    Takes per-modality Dirichlet alphas as input and outputs fused belief masses
    and uncertainty.

    Alphas are [B, N, K+1] where K+1 includes explicit background alpha.
    For MM-UAV (K=1 drone): alphas[:, 0] = softplus(logit)+1, alphas[:, 1] = 1.
    """

    def __init__(self, num_classes, C_max=0.95):
        super().__init__()
        self.num_classes = num_classes     # K (foreground classes, typically 1)
        self.dirichlet_dim = num_classes + 1  # K+1 (includes bg)
        self.C_max = C_max

    def forward(self, alphas_rgb, alphas_ir):
        """Fuse two modalities.

        Args:
            alphas_rgb: shape [B, N, K+1] Dirichlet parameters from RGB head
            alphas_ir:  shape [B, N, K+1] Dirichlet parameters from IR head

        Returns:
            b_fused: shape [B, N, K+1] fused belief masses
            u_fused: shape [B, N, 1]   fused uncertainty
            C:       shape [B, N, 1]   conflict
            fallback: shape [B, N, 1]  bool
        """
        B, N, _ = alphas_rgb.shape
        alphas = torch.stack([alphas_rgb, alphas_ir], dim=2)  # [B, N, 2, K+1]
        b, u = alpha_to_bpa(alphas)  # [B, N, 2, K+1], [B, N, 2, 1]
        return ds_combine(b[:, :, 0], u[:, :, 0], b[:, :, 1], u[:, :, 1], self.C_max)
