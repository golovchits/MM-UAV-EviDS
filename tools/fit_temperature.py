#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# E3 Condition (g): Fit temperature scaling on baseline (a) checkpoint.
#
# Collects raw obj+cls logits on a tune split (120 held-out train sequences),
# then optimises T by minimising NLL of GT-matched anchors.
#
# Usage: python tools/fit_temperature.py
# Output: YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.exp import get_exp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TUNE_SEQ = 120


def box_iou(boxes1, boxes2):
    """boxes: [N, 4] (x1, y1, x2, y2)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def make_tune_loaders(exp):
    """Create eval-style loaders filtered to first N_TUNE_SEQ training sequences."""
    from pycocotools.coco import COCO

    def vid_key(img):
        # video_id may be int or absent; fall back to first dir component of file_name
        v = img.get("video_id")
        if v is not None:
            return str(v)
        return img.get("file_name", "").split('/')[0]

    def filter_json(src_json, dst_json, tune_video_ids):
        with open(src_json) as f:
            data = json.load(f)
        vid_set = set(tune_video_ids)
        tune_img_ids = {img["id"] for img in data["images"] if vid_key(img) in vid_set}
        data["images"] = [img for img in data["images"] if img["id"] in tune_img_ids]
        keep_img_ids = {img["id"] for img in data["images"]}
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep_img_ids]
        with open(dst_json, "w") as f:
            json.dump(data, f)

    train_json = os.path.join(exp.data_dir, "annotations", exp.train_ann1)
    coco = COCO(train_json)
    all_vids = sorted({vid_key(img) for img in coco.imgs.values()})
    tune_vids = all_vids[:N_TUNE_SEQ]
    print(f"Tune split: {len(tune_vids)} sequences ({tune_vids[0]}..{tune_vids[-1]})")

    tune_dir = os.path.join(exp.data_dir, "annotations")
    filter_json(train_json, os.path.join(tune_dir, "tune-rgb.json"), tune_vids)
    filter_json(os.path.join(exp.data_dir, "annotations", exp.train_ann2),
                os.path.join(tune_dir, "tune-ir.json"), tune_vids)

    exp.val_ann1 = "tune-rgb.json"
    exp.val_ann2 = "tune-ir.json"
    return exp.get_eval_loader(batch_size=8, is_distributed=False)


def collect_logits(model, loader1, loader2, test_size=(640, 640), candidate_thresh=0.1):
    """Run inference and collect (obj_logit, cls_logit, is_positive) per anchor.

    Only considers candidate anchors (sigmoid(obj_logit) > candidate_thresh).
    Calibrating on all ~8400 anchors fails because 99%+ are background — T stays at 1.0.
    Standard detector calibration restricts to candidates the model "noticed."

    Positives: IoU >= 0.5 against GT, using the model's own decoded boxes.
    GT boxes from COCO (original image coords) are scaled by preprocessing ratio r
    to match the model's output coordinate space (resized+padded input space).
    """
    obj_logits_acc, cls_logits_acc, positives_acc = [], [], []

    for (imgs1, _, infos1, ids1), (imgs2, _, infos2, ids2) in zip(
            tqdm(loader1, desc="Collecting"), loader2):
        imgs1, imgs2 = imgs1.to(DEVICE), imgs2.to(DEVICE)

        with torch.no_grad():
            out1, out2 = model(imgs1, imgs2)  # [B, N, 5+K] pre-NMS decoded, input coords

        for head, out, loader, imgs, infos, ids in [
            (model.head,  out1, loader1, imgs1, infos1, ids1),
            (model.head2, out2, loader2, imgs2, infos2, ids2),
        ]:
            obj_logit = head._last_obj_logits  # list of [B, 1, H, W] per FPN level
            cls_logit = head._last_cls_logits  # list of [B, K, H, W] per FPN level

            # Flatten in FPN decode order — matches out[b] anchor ordering
            obj_flat = torch.cat([o.flatten(2) for o in obj_logit], dim=2)  # [B, 1, N]
            cls_flat = torch.cat([c.flatten(2) for c in cls_logit], dim=2)  # [B, K, N]
            obj_flat = obj_flat.permute(0, 2, 1).squeeze(-1)   # [B, N]
            cls_flat = cls_flat.permute(0, 2, 1)               # [B, N, K]

            for b in range(imgs.shape[0]):
                img_id = int(ids[b].item()) if torch.is_tensor(ids[b]) else int(ids[b])
                img_h, img_w = int(infos[0][b].item()), int(infos[1][b].item())

                # Restrict to candidate anchors — obj > threshold (raw logit threshold)
                cand_mask = torch.sigmoid(obj_flat[b]) > candidate_thresh  # [N]
                if cand_mask.sum() == 0:
                    continue

                obj_cand = obj_flat[b][cand_mask].cpu()   # [C]
                cls_cand = cls_flat[b][cand_mask].cpu()   # [C, K]

                ann_ids = loader.dataset.coco.getAnnIds(imgIds=img_id)
                anns = loader.dataset.coco.loadAnns(ann_ids)
                if not anns:
                    # No GT — all candidates are negatives; still useful for calibration
                    positives_acc.append(np.zeros(int(cand_mask.sum()), dtype=bool))
                    obj_logits_acc.append(obj_cand.numpy())
                    cls_logits_acc.append(cls_cand[:, 0].numpy())
                    continue

                # GT boxes: COCO xywh → x1y1x2y2, original image coords
                gt_boxes = torch.tensor(
                    [[a['bbox'][0], a['bbox'][1],
                      a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                     for a in anns], dtype=torch.float32
                )
                # Scale GT to match model output coords (same ratio as preprocessing)
                r = min(test_size[0] / img_h, test_size[1] / img_w)
                gt_boxes_scaled = gt_boxes * r

                # Model's decoded output: cx, cy, w, h → x1, y1, x2, y2
                pred_cxcywh = out[b, :, :4][cand_mask].cpu()  # [C, 4]
                pred_boxes = torch.stack([
                    pred_cxcywh[:, 0] - pred_cxcywh[:, 2] / 2,
                    pred_cxcywh[:, 1] - pred_cxcywh[:, 3] / 2,
                    pred_cxcywh[:, 0] + pred_cxcywh[:, 2] / 2,
                    pred_cxcywh[:, 1] + pred_cxcywh[:, 3] / 2,
                ], dim=1)  # [C, 4]

                ious = box_iou(pred_boxes, gt_boxes_scaled)  # [C, G]
                max_ious, _ = ious.max(dim=1)
                is_pos = max_ious >= 0.5

                obj_logits_acc.append(obj_cand.numpy())
                cls_logits_acc.append(cls_cand[:, 0].numpy())  # K=1
                positives_acc.append(is_pos.numpy())

    return (np.concatenate(obj_logits_acc),
            np.concatenate(cls_logits_acc),
            np.concatenate(positives_acc))


def compute_ece(obj_logits, cls_logits, positives, T, n_bins=15):
    """Expected Calibration Error for binary detection confidence."""
    scores = (torch.sigmoid(torch.tensor(obj_logits, dtype=torch.float32) / T) *
              torch.sigmoid(torch.tensor(cls_logits, dtype=torch.float32) / T))
    is_pos = torch.tensor(positives, dtype=torch.float32)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        conf = scores[mask].mean().item()
        acc = is_pos[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(conf - acc)
    return ece


def fit_temperature(obj_logits, cls_logits, positives):
    """Optimise T via L-BFGS to minimise NLL of temperature-scaled scores."""
    obj_t = torch.tensor(obj_logits, dtype=torch.float32)
    cls_t = torch.tensor(cls_logits, dtype=torch.float32)
    pos_t = torch.tensor(positives, dtype=torch.float32)

    T = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        p = torch.sigmoid(obj_t / T) * torch.sigmoid(cls_t / T)
        p = p.clamp(1e-7, 1 - 1e-7)
        loss = -(pos_t * torch.log(p) + (1 - pos_t) * torch.log(1 - p)).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    T_val = T.item()
    print(f"Fitted T = {T_val:.4f}")
    return T_val


def main():
    exp_file = "yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py"
    ckpt_path = "YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/best_ckpt.pth.tar"

    print(f"Loading {exp_file}")
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print("Model loaded.")

    model.head.T = 1.0
    model.head2.T = 1.0

    loader1, loader2 = make_tune_loaders(exp)
    test_size = tuple(exp.test_size)

    obj_logits, cls_logits, positives = collect_logits(
        model, loader1, loader2, test_size=test_size, candidate_thresh=0.1)

    n_pos = int(positives.sum())
    n_total = len(positives)
    pct = n_pos / n_total * 100
    print(f"Collected {n_total:,} anchors ({n_pos:,} positive, {pct:.2f}%)")
    if pct < 1.0:
        print("WARNING: positive fraction <1% — NLL dominated by background. "
              "T will calibrate background confidence, not detection. "
              "Check GT matching (coordinate space, IoU threshold).")

    ece_before = compute_ece(obj_logits, cls_logits, positives, T=1.0)

    T = fit_temperature(obj_logits, cls_logits, positives)
    if not (0.3 <= T <= 5.0):
        print(f"WARNING: T={T:.4f} outside expected range [0.3, 5.0] — check positive fraction.")

    ece_after = compute_ece(obj_logits, cls_logits, positives, T=T)
    print(f"ECE before T-scaling: {ece_before:.4f}")
    print(f"ECE after  T-scaling: {ece_after:.4f}")
    if ece_after >= ece_before:
        print("WARNING: T-scaling did not improve ECE. "
              "If positive fraction is <1%, restrict to obj>0.1 anchors.")

    out_dir = "YOLOX_outputs/yolox_s_2_def-tuning-fusion-head"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "temperature.json")
    with open(out_path, "w") as f:
        json.dump({
            "T": T,
            "n_anchors": n_total,
            "n_positive": n_pos,
            "positive_pct": round(pct, 3),
            "ece_before": round(ece_before, 6),
            "ece_after": round(ece_after, 6),
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
