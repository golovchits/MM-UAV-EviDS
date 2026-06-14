#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EviDS-UAV: Temporal Sensor Gating (Eq. 6 in methodology)
#
# Hysteresis gate: disable a modality when its epistemic uncertainty u
# exceeds tau_high for N consecutive frames. Re-enable when u drops below
# tau_low. Constraint: at least one modality always active.

import torch
import torch.nn as nn


class TemporalGate(nn.Module):
    """Hysteresis gate per modality (Eq. 6).

    Tracks per-modality uncertainty streaks. When u > tau_high for N
    consecutive frames, the modality is gated (suppressed) for subsequent
    frames until u < tau_low clears the gate.

    Constraint: caller ensures at least one modality stays active.
    """

    def __init__(self, tau_high=0.65, tau_low=0.40, N=8, num_modalities=2):
        """
        Args:
            tau_high: gate fires when u > tau_high for N consecutive frames
            tau_low: gate clears when u < tau_low
            N: consecutive frames required before gate activates
            num_modalities: number of sensor modalities (default 2: RGB + IR)
        """
        super().__init__()
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.N = N
        self.num_modalities = num_modalities

        # Per-modality state (not registered as buffers since they change dynamically)
        self.register_buffer('streak', torch.zeros(num_modalities))
        self.register_buffer('active_gate', torch.zeros(num_modalities, dtype=torch.bool))

    def reset(self):
        """Reset gate state (call between sequences)."""
        self.streak.zero_()
        self.active_gate.zero_()

    def step(self, u_per_modality):
        """Update gate state given per-modality uncertainty.

        Args:
            u_per_modality: tensor of shape [M] with epistemic uncertainties

        Returns:
            gated: bool tensor [M] — True = modality is gated (suppressed)
        """
        M = u_per_modality.shape[0]
        for m in range(M):
            u = u_per_modality[m].item()
            if u > self.tau_high:
                self.streak[m] += 1
                if self.streak[m] >= self.N:
                    self.active_gate[m] = True
            elif u < self.tau_low:
                self.streak[m] = 0
                self.active_gate[m] = False
            # else: in hysteresis band, hold current state

        return self.active_gate.clone()

    def forward(self, u_rgb, u_ir):
        """Convenience wrapper for 2-modality case (RGB + IR).

        Args:
            u_rgb: scalar uncertainty from RGB head
            u_ir: scalar uncertainty from IR head

        Returns:
            gate_rgb: bool, True = gated
            gate_ir: bool, True = gated
        """
        u = torch.tensor([u_rgb.item() if torch.is_tensor(u_rgb) else u_rgb,
                          u_ir.item() if torch.is_tensor(u_ir) else u_ir],
                         device=self.streak.device)
        gated = self.step(u)

        # Enforce constraint: at least one modality active
        if gated.all():
            # Keep modality with lower uncertainty active
            u_vals = u
            best = u_vals.argmin().item()
            gated[best] = False

        return gated[0].item(), gated[1].item()
