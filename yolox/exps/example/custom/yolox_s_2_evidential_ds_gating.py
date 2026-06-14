#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EviDS-UAV Experiment 3 — Condition (f): Evidential + DS + Gating
#
# Full EviDS-UAV. Evidential heads + DS fusion + temporal sensor gating.
# OGAA spatial alignment retained, ADFM removed.
# Hysteresis gate disables RGB or IR when epistemic uncertainty
# exceeds tau_high for N consecutive frames; re-enables below tau_low.
# Constraint: at least one modality always active.

import os
import torch
import torch.nn as nn

from yolox.exp import Exp2 as MyExp2


class Exp(MyExp2):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = os.environ.get("DATA_DIR", "/path/to/MM-UAV-images/")
        self.use_tar = False
        self.train_ann1 = "train-rgb.json"
        self.val_ann1 = "val-rgb.json"
        self.train_ann2 = "train-ir.json"
        self.val_ann2 = "val-ir.json"

        self.num_classes = 1
        self.max_epoch = 50
        self.data_num_workers = 0
        self.eval_interval = 1
        self.no_aug_epochs = 50
        self.enable_mixup = False
        self.random_size = None

        self.kl_anneal_epochs = 10
        self.ds_C_max = 0.95

        # Temporal gating hyperparameters (from demo calibration)
        self.tau_high = 0.65
        self.tau_low = 0.40
        self.gate_N = 8  # consecutive frames

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

            optimizer = torch.optim.AdamW(
                [
                    {"params": pg0, "weight_decay": 0.0},
                    {"params": pg1, "weight_decay": self.weight_decay},
                    {"params": pg2, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=(0.9, 0.999),
            )
            self.optimizer = optimizer
        return self.optimizer

    def get_model(self):
        from yolox.models import YOLOX2DS
        from yolox.models.yolo_head_evidential import YOLOXHeadEvidential
        from yolox.models.yolo_pafpn2_evidential_ds import YOLOPAFPN2
        from yolox.models.ds_fusion import DSFusion
        from yolox.models.temporal_gate import TemporalGate

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN2(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHeadEvidential(
                self.num_classes, self.width, in_channels=in_channels,
                kl_anneal_epochs=self.kl_anneal_epochs
            )
            head2 = YOLOXHeadEvidential(
                self.num_classes, self.width, in_channels=in_channels,
                kl_anneal_epochs=self.kl_anneal_epochs
            )
            ds_fusion = DSFusion(num_classes=self.num_classes, C_max=self.ds_C_max)
            temporal_gate = TemporalGate(
                tau_high=self.tau_high,
                tau_low=self.tau_low,
                N=self.gate_N,
                num_modalities=2,
            )
            self.model = YOLOX2DS(backbone, head, head2, ds_fusion, temporal_gate)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        weight_path = "YOLOX_outputs/yolox_s_2_stream/best_ckpt.pth.tar"
        if not os.path.exists(weight_path):
            raise RuntimeError("No pretrained 1 stage weights found at {}".format(weight_path))

        checkpoint = torch.load(weight_path, map_location="cpu")
        pretrained_state_dict = checkpoint['model']
        model_state_dict = self.model.state_dict()

        matched_keys = []
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                model_state_dict[k] = v
                matched_keys.append(k)

        self.model.load_state_dict(model_state_dict, strict=False)

        for name, param in self.model.named_parameters():
            if 'Fusion0' not in name and 'Fusion1' not in name and 'Fusion2' not in name:
                if 'head' not in name:
                    param.requires_grad = False

        return self.model
