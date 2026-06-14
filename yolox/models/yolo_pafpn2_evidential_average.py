#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# EviDS-UAV: Evidential FPN for Dirichlet averaging (condition d).
#
# Uses the SAME OGAA-only FPN as condition (e) — OGAA spatial alignment
# retained, no ADFM feature fusion. The difference between (d) and (e) is
# at the decision level: (d) averages Dirichlet class probabilities p_k,
# while (e) uses DS evidence combination. This isolates the fusion method
# as the sole variable.

from .yolo_pafpn2_evidential_ds import YOLOPAFPN2  # noqa: F401
