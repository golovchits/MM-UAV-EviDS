#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# EviDS-UAV: Evidential FPN with ADFM retained (condition c).
#
# Identical to yolo_pafpn2_def.py — ADFM (channel-attention fusion) is
# retained. The evidential heads operate on ADFM-fused features.
# Each stream independently outputs Dirichlet parameters.

from .yolo_pafpn2_def import YOLOPAFPN2  # noqa: F401
