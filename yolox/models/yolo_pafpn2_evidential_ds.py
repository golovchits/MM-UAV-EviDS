#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# EviDS-UAV: Evidential FPN with OGAA only (no ADFM, condition e/f).
#
# Retains OGAA spatial alignment (deformable convolution) but removes
# ADFM channel-attention fusion. Each stream receives aligned features
# independently. Evidential heads produce per-modality Dirichlet outputs,
# and DS evidence fusion happens at the decision level (in the model wrapper).

import torch
import torch.nn as nn
import torchvision

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


# ── OGAA modules (identical to yolo_pafpn2_def.py) ──────────────────────────

class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, padding=9, dilation=9)
        self.fuse = nn.Conv2d(4 * out_ch, out_ch, 1)

    def forward(self, x):
        return self.fuse(torch.cat([
            self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        ], dim=1))


class Offset_Module(nn.Module):
    def __init__(self, c1, k=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.aspp1 = ASPPModule(8, 8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.leakyReLU2 = nn.LeakyReLU(0.2)
        self.conv_offset = nn.Conv2d(8, 2 * k * k, kernel_size=1)

    def forward(self, feature):
        x = self.leakyReLU(self.bn1(self.conv1(feature)))
        x = self.aspp1(x)
        x = self.leakyReLU2(self.bn2(self.conv2(x)))
        offset = self.conv_offset(x)
        return offset


class Align_DefConv(nn.Module):
    def __init__(self, c1, k=1):
        super().__init__()
        self.offset_conv = Offset_Module(16, k=k)
        self.weight = torch.ones(c1, 1, k, k)

    def forward(self, x, feature_for_offset):
        offset = self.offset_conv(feature_for_offset)
        if self.weight.device != x.device or self.weight.dtype != x.dtype:
            weight = self.weight.to(device=x.device, dtype=x.dtype)
        else:
            weight = self.weight
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=weight)


# ── OGAA-only "fusion" (alignment, no cross-modal combination) ──────────────

class Fusion(nn.Module):
    """OGAA alignment only. No feature-level fusion.

    Each stream gets its own feature enhanced by alignment with the other
    modality's features, but no cross-modal feature combination occurs.
    DS evidence fusion at the decision level handles combination.
    """

    def __init__(self, c1):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1)
        self.align_for_rgb = Align_DefConv(c1)
        self.align_for_ir = Align_DefConv(c1)

    def forward(self, x_rgb, x_ir):
        x_rgb_feat = self.conv1(x_rgb)
        x_ir_feat = self.conv1(x_ir)

        # OGAA: align each modality using both as input
        x_rgb_offset = self.align_for_rgb(
            x_rgb, torch.cat([x_rgb_feat, x_ir_feat], dim=1))
        x_ir_offset = self.align_for_ir(
            x_ir, torch.cat([x_rgb_feat, x_ir_feat], dim=1))

        # Residual connection only — no cross-modal fusion
        x_rgb_out = x_rgb + x_rgb_offset
        x_ir_out = x_ir + x_ir_offset
        return x_rgb_out, x_ir_out


# ── YOLOPAFPN2 (same structure as def variant) ──────────────────────────────

class YOLOPAFPN2(nn.Module):
    """Dual-stream FPN with OGAA alignment only (no feature-level fusion)."""

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone_rgb = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.backbone_ir = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # ── RGB path ──
        self.lateral_conv0_rgb = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4_rgb = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1_rgb = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3_rgb = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2_rgb = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3_rgb = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1_rgb = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4_rgb = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

        # ── IR path ──
        self.lateral_conv0_ir = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4_ir = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1_ir = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3_ir = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2_ir = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3_ir = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1_ir = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4_ir = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

        self.Fusion0 = Fusion(int(in_channels[2] * width))
        self.Fusion1 = Fusion(int(in_channels[1] * width))
        self.Fusion2 = Fusion(int(in_channels[0] * width))

    def forward(self, input1, input2):
        out_features1 = self.backbone_rgb(input1)
        out_features2 = self.backbone_ir(input2)
        features1 = [out_features1[f] for f in self.in_features]
        features2 = [out_features2[f] for f in self.in_features]
        [x2_rgb, x1_rgb, x0_rgb] = features1
        [x2_ir, x1_ir, x0_ir] = features2

        x0_rgb, x0_ir = self.Fusion0(x0_rgb, x0_ir)
        x1_rgb, x1_ir = self.Fusion1(x1_rgb, x1_ir)
        x2_rgb, x2_ir = self.Fusion2(x2_rgb, x2_ir)

        # ── RGB FPN + PAN ──
        fpn_out0_rgb = self.lateral_conv0_rgb(x0_rgb)
        f_out0_rgb = self.upsample(fpn_out0_rgb)
        f_out0_rgb = torch.cat([f_out0_rgb, x1_rgb], 1)
        f_out0_rgb = self.C3_p4_rgb(f_out0_rgb)
        fpn_out1_rgb = self.reduce_conv1_rgb(f_out0_rgb)
        f_out1_rgb = self.upsample(fpn_out1_rgb)
        f_out1_rgb = torch.cat([f_out1_rgb, x2_rgb], 1)
        pan_out2_rgb = self.C3_p3_rgb(f_out1_rgb)
        p_out1_rgb = self.bu_conv2_rgb(pan_out2_rgb)
        p_out1_rgb = torch.cat([p_out1_rgb, fpn_out1_rgb], 1)
        pan_out1_rgb = self.C3_n3_rgb(p_out1_rgb)
        p_out0_rgb = self.bu_conv1_rgb(pan_out1_rgb)
        p_out0_rgb = torch.cat([p_out0_rgb, fpn_out0_rgb], 1)
        pan_out0_rgb = self.C3_n4_rgb(p_out0_rgb)

        # ── IR FPN + PAN ──
        fpn_out0_ir = self.lateral_conv0_ir(x0_ir)
        f_out0_ir = self.upsample(fpn_out0_ir)
        f_out0_ir = torch.cat([f_out0_ir, x1_ir], 1)
        f_out0_ir = self.C3_p4_ir(f_out0_ir)
        fpn_out1_ir = self.reduce_conv1_ir(f_out0_ir)
        f_out1_ir = self.upsample(fpn_out1_ir)
        f_out1_ir = torch.cat([f_out1_ir, x2_ir], 1)
        pan_out2_ir = self.C3_p3_ir(f_out1_ir)
        p_out1_ir = self.bu_conv2_ir(pan_out2_ir)
        p_out1_ir = torch.cat([p_out1_ir, fpn_out1_ir], 1)
        pan_out1_ir = self.C3_n3_ir(p_out1_ir)
        p_out0_ir = self.bu_conv1_ir(pan_out1_ir)
        p_out0_ir = torch.cat([p_out0_ir, fpn_out0_ir], 1)
        pan_out0_ir = self.C3_n4_ir(p_out0_ir)

        outputs1 = (pan_out2_rgb, pan_out1_rgb, pan_out0_rgb)
        outputs2 = (pan_out2_ir, pan_out1_ir, pan_out0_ir)
        return outputs1, outputs2
