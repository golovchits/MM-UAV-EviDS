#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolo_pafpn2 import YOLOPAFPN2

from .yolo_pafpn2_stn_noFusion import YOLOPAFPN2 as YOLOPAFPN2_stn_noFusion
from .yolo_pafpn2_def_noFusion import YOLOPAFPN2 as YOLOPAFPN2_def_noFusion

from .yolo_pafpn2_def import YOLOPAFPN2 as YOLOPAFPN2_def
from .yolo_pafpn2_stn import YOLOPAFPN2 as YOLOPAFPN2_stn

from .yolox import YOLOX
from .yolox2 import YOLOX2
from .yolox2_ds import YOLOX2DS
from .yolox2_average import YOLOX2Average

from .yolo_head_evidential import YOLOXHeadEvidential
from .ds_fusion import DSFusion
from .temporal_gate import TemporalGate

from .yolo_pafpn2_evidential_adfm import YOLOPAFPN2 as YOLOPAFPN2_evidential_adfm
from .yolo_pafpn2_evidential_average import YOLOPAFPN2 as YOLOPAFPN2_evidential_average
from .yolo_pafpn2_evidential_ds import YOLOPAFPN2 as YOLOPAFPN2_evidential_ds
