#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
E3 Balanced-Crop Experiment — Step 1: Extract Detection Crops
=============================================================

Runs the full EDL YOLOX detection pipeline on the MM-UAV test set and, for each
post-NMS detection, extracts the image crop around the bounding box with context
padding. Saves per-detection metadata (bbox, vacuity, p_uav, TP/FP label) and
the cropped image patches for downstream balanced-classification evaluation.

This is the first step of the within-experiment positive control that isolates
class imbalance as the cause of vacuity inversion at the detection level.

Outputs:
  tools/e3_crops/
    metadata.json          — per-detection records (bbox, vacuity, TP/FP, crop path)
    crops/
      {cond}_{img_id:06d}_{det_idx:04d}.png  — cropped image patches

Usage on Snellius:
    module purge && module load 2023
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
    source /gpfs/work5/0/prjs1970/envs/mm-uav-venv/bin/activate

    python tools/extract_detection_crops.py --cond d
    python tools/extract_detection_crops.py --cond d --max-images 50  # quick test
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Add MM-UAV-Benchmark root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.models.yolo_head_evidential import softplus_evidence


# ── Configuration ──────────────────────────────────────────────────────────

# Conditions that have evidential heads and are relevant for the experiment
CONDITIONS = {
    "d": {
        "label": "Evidential + Average",
        "exp_file": "yolox/exps/example/custom/yolox_s_2_evidential_average.py",
        "ckpt": "YOLOX_outputs/yolox_s_2_evidential_average/best_ckpt.pth.tar",
    },
    "e": {
        "label": "Evidential + DS",
        "exp_file": "yolox/exps/example/custom/yolox_s_2_evidential_ds.py",
        "ckpt": "YOLOX_outputs/yolox_s_2_evidential_ds/best_ckpt.pth.tar",
    },
}

# Detection threshold: use the analysis floor (0.001) to capture ALL possible
# detections — we want the full distribution, not just confident ones.
CONF_THRESHOLD = 0.001
NMS_THRESHOLD = 0.65

# Crop parameters: expand bbox by this factor on each side to include context
CROP_CONTEXT_FACTOR = 1.5   # expand bbox by 1.5× in each dimension
CROP_MIN_SIZE = 64           # minimum crop size (pixels) after expansion
CROP_TARGET_SIZE = (640, 640)  # resize crops to this for downstream inference

# Output directory — use scratch to avoid work5 inode exhaustion.
# Override with --out-dir.
OUTPUT_DIR = os.environ.get("CROP_DIR", "/path/to/e3_crops")


# ── Helpers ─────────────────────────────────────────────────────────────────

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes. Both in [x1, y1, x2, y2] format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def extract_crop(image, bbox_xyxy, context_factor=1.5, min_size=64):
    """Extract a square crop around a bounding box with context padding.

    Args:
        image: numpy array [H, W, 3] in BGR (OpenCV format)
        bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
        context_factor: expand bbox dimensions by this factor
        min_size: minimum crop side length in pixels

    Returns:
        crop: square numpy array [min_size, min_size, 3] or larger, or None if invalid
    """
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]

    # Compute center and size with context expansion
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * context_factor, (y2 - y1) * context_factor
    side = max(bw, bh, min_size)

    # Compute crop boundaries
    crop_x1 = int(cx - side / 2)
    crop_y1 = int(cy - side / 2)
    crop_x2 = int(cx + side / 2)
    crop_y2 = int(cy + side / 2)

    # Pad with black (0) if crop extends outside image
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    # Clamp to valid image range
    valid_x1 = max(0, crop_x1)
    valid_y1 = max(0, crop_y1)
    valid_x2 = min(w, crop_x2)
    valid_y2 = min(h, crop_y2)

    if valid_x2 <= valid_x1 or valid_y2 <= valid_y1:
        return None

    crop = image[valid_y1:valid_y2, valid_x1:valid_x2]
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    if crop.shape[0] < 1 or crop.shape[1] < 1:
        return None

    return crop


def resize_crop(crop, target_size=(640, 640)):
    """Resize crop to target size, preserving aspect ratio by padding."""
    h, w = crop.shape[:2]
    r = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size (top-left aligned, pad bottom/right with 0)
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    return padded


def evidential_stats(cls_logits):
    """Compute Dirichlet p_uav and vacuity from per-anchor classification logits.

    For K=1 (single-class) with implicit alpha_bg=1:
      alpha_drone = softplus(logit) + 1
      S_eff = alpha_drone + 1    (alpha_bg=1 implicit)
      p_uav = alpha_drone / S_eff
      u = 1 / S_eff

    Args:
        cls_logits: tensor [B, N, K] raw classification logits

    Returns:
        p_uav: [B, N] predicted UAV probability per anchor
        u: [B, N] epistemic vacuity per anchor
    """
    evidence = softplus_evidence(cls_logits)
    alphas = evidence + 1.0
    S_eff = alphas.sum(dim=-1, keepdim=True) + 1.0   # +1 for implicit alpha_bg=1
    p = (alphas / S_eff).squeeze(-1)                   # [B, N]
    u = 1.0 / S_eff.squeeze(-1)                        # [B, N] — K=1, so K/S_eff
    return p, u


def match_det_vacuity(decoded_out, nms_dets, u_per_anchor):
    """Match each post-NMS detection to its pre-NMS anchor vacuity via IoU.

    Args:
        decoded_out: [N, 5+K] model decoded output (cx,cy,w,h,...) in input space
        nms_dets: [D, 7] post-NMS detections (x1,y1,x2,y2,obj,cls,cls_idx) or None
        u_per_anchor: [N] vacuity tensor

    Returns:
        det_u: [D] vacuity for each post-NMS detection
        det_p: [D] p_uav for each post-NMS detection
        det_boxes: [D, 4] bbox in (x1,y1,x2,y2) input-space coordinates
    """
    if nms_dets is None or len(nms_dets) == 0:
        return np.array([]), np.array([]), np.zeros((0, 4))

    # Convert pre-NMS anchor boxes from cx,cy,w,h to x1,y1,x2,y2
    pre = decoded_out[:, :4].cpu()
    pre_corners = torch.stack([
        pre[:, 0] - pre[:, 2] / 2, pre[:, 1] - pre[:, 3] / 2,
        pre[:, 0] + pre[:, 2] / 2, pre[:, 1] + pre[:, 3] / 2,
    ], dim=1)

    det_boxes = nms_dets[:, :4].cpu()

    # IoU matching
    area1 = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
    area2 = (pre_corners[:, 2] - pre_corners[:, 0]) * (pre_corners[:, 3] - pre_corners[:, 1])
    lt = torch.max(det_boxes[:, None, :2], pre_corners[None, :, :2])
    rb = torch.min(det_boxes[:, None, 2:], pre_corners[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    ious = inter / (union + 1e-8)
    best_idx = ious.argmax(dim=1)  # [D]

    return (u_per_anchor[best_idx].cpu().numpy(),
            det_boxes.cpu().numpy())


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_crops_for_condition(cond_key, cond_info, device, max_images=None,
                               out_root=None):
    """Run detection inference and extract crops for one condition."""
    if out_root is None:
        out_root = OUTPUT_DIR
    label = cond_info["label"]
    exp_file = os.path.join(ROOT, cond_info["exp_file"])
    ckpt_path = os.path.join(ROOT, cond_info["ckpt"])

    out_dir = os.path.join(out_root, cond_key)
    crop_dir = os.path.join(out_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    # Pre-check: verify the output directory is actually writable before
    # spending GPU hours on detection inference.
    test_file = os.path.join(crop_dir, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except OSError as e:
        print(f"FATAL: Cannot write to {crop_dir}: {e}")
        print("Check disk quota (myquota) and filesystem.")
        sys.exit(1)
    print(f"  Write test passed: {crop_dir}")

    print(f"\n{'='*70}")
    print(f"Condition ({cond_key}): {label}")
    print(f"  Exp: {exp_file}")
    print(f"  Ckpt: {ckpt_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Load experiment and model
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"  Model loaded. Best epoch: {ckpt.get('start_epoch', '?')}")

    # Get evaluation data loaders
    val_loader1, val_loader2 = exp.get_eval_loader(batch_size=8, is_distributed=False)
    num_classes = exp.num_classes
    print(f"  Test set: RGB={len(val_loader1.dataset)}, IR={len(val_loader2.dataset)} images")

    # Per-detection records
    all_records = []
    crop_idx = 0
    t_start = time.time()

    for batch_idx, ((imgs1, _, infos1, ids1), (imgs2, _, infos2, ids2)) in enumerate(
        tqdm(zip(val_loader1, val_loader2), total=len(val_loader1), desc="Detecting + cropping")
    ):
        if max_images and batch_idx * 8 >= max_images:
            break

        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)

        with torch.no_grad():
            # Run full model — returns (fused_output, raw_rgb_output)
            outputs1, outputs2 = model(imgs1, imgs2)

        # Get per-head cls_logits for vacuity computation
        cls_logits1 = getattr(model.head, '_last_cls_logits', None)
        cls_logits2 = getattr(model.head2, '_last_cls_logits', None)

        # Batch NMS via postprocess
        dets1 = postprocess(outputs1.clone(), num_classes, conf_thre=CONF_THRESHOLD,
                            nms_thre=NMS_THRESHOLD)
        dets2 = postprocess(outputs2.clone(), num_classes, conf_thre=CONF_THRESHOLD,
                            nms_thre=NMS_THRESHOLD)

        for b in range(imgs1.shape[0]):
            img_id = int(ids1[b].item()) if torch.is_tensor(ids1[b]) else int(ids1[b])
            img_h, img_w = int(infos1[0][b].item()), int(infos1[1][b].item())

            # Get ground truth annotations for this image
            ann_ids = val_loader1.dataset.coco.getAnnIds(imgIds=img_id)
            anns = val_loader1.dataset.coco.loadAnns(ann_ids)
            gt_boxes_xyxy = np.array([
                [a['bbox'][0], a['bbox'][1],
                 a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                for a in anns
            ], dtype=np.float32) if anns else np.zeros((0, 4))

            # ── Process RGB stream ────────────────────────────────────────
            pred = dets1[b]
            if pred is not None and pred.shape[0] > 0:
                boxes = pred[:, :4].cpu()  # [D, 4] in pre-NMS output space (input space)
                scores = (pred[:, 4] * pred[:, 5]).cpu()  # obj_conf * cls_conf

                # Scale boxes from input space back to original image space
                scale = min(exp.test_size[0] / img_h, exp.test_size[1] / img_w)
                boxes_orig = boxes.clone()
                boxes_orig[:, :4] = boxes[:, :4] / scale

                # Match detections to GT via IoU for TP/FP labeling
                if len(gt_boxes_xyxy) > 0:
                    ious = box_iou(
                        boxes_orig.numpy().astype(np.float32),
                        gt_boxes_xyxy.astype(np.float32)
                    )
                    max_ious = ious.max(axis=1)
                    is_tp = max_ious >= 0.5
                else:
                    is_tp = np.zeros(len(boxes), dtype=bool)

                # Match each post-NMS detection to its pre-NMS anchor vacuity
                if cls_logits1 is not None:
                    p_uav_per_anchor, u_per_anchor = evidential_stats(cls_logits1[b:b+1])
                    # Use decoded outputs (pre-NMS) from the model's head
                    # outputs2 is raw RGB decoded output before DS/averaging
                    decoded = outputs2[b]  # [N, 5+K] decoded
                    det_u, det_boxes_np = match_det_vacuity(
                        decoded, pred, u_per_anchor[0]
                    )
                    det_p_uav, _ = match_det_vacuity(
                        decoded, pred, p_uav_per_anchor[0]
                    )
                else:
                    det_u = np.zeros(len(boxes), dtype=np.float32)
                    det_p_uav = scores.numpy()

                # Load raw (un-preprocessed) image for crop extraction
                # Use the COCO API to look up the image file path from image ID
                raw_img = None
                try:
                    coco = val_loader1.dataset.coco
                    img_info = coco.loadImgs([img_id])[0]
                    file_name = img_info['file_name']
                    img_path = os.path.join(
                        val_loader1.dataset.data_dir,
                        val_loader1.dataset.name,
                        file_name,
                    )
                    raw_img = cv2.imread(img_path)
                except Exception:
                    pass

                # For each detection, extract crop and record metadata
                for d in range(len(boxes)):
                    if max_images and len(all_records) >= max_images * 10:
                        break

                    x1, y1, x2, y2 = boxes_orig[d].numpy().astype(np.float32)
                    # Clamp to image bounds
                    x1 = max(0, min(img_w - 1, x1))
                    y1 = max(0, min(img_h - 1, y1))
                    x2 = max(x1 + 1, min(img_w, x2))
                    y2 = max(y1 + 1, min(img_h, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    record = {
                        "crop_id": crop_idx,
                        "cond": cond_key,
                        "img_id": img_id,
                        "det_idx": d,
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "p_uav_orig": float(det_p_uav[d]) if len(det_p_uav) > d else float(scores[d]),
                        "vacuity_orig": float(det_u[d]) if len(det_u) > d else np.nan,
                        "confidence_orig": float(scores[d]),
                        "is_tp": bool(is_tp[d]),
                        "max_iou": float(max_ious[d]) if len(gt_boxes_xyxy) > 0 else 0.0,
                        "crop_path": os.path.join("crops", f"{cond_key}_{crop_idx:06d}.png"),
                    }

                    all_records.append(record)

                    # Extract and save crop
                    if raw_img is not None:
                        crop = extract_crop(raw_img, [x1, y1, x2, y2],
                                           context_factor=CROP_CONTEXT_FACTOR,
                                           min_size=CROP_MIN_SIZE)
                        if crop is not None:
                            crop_resized = resize_crop(crop, CROP_TARGET_SIZE)
                            crop_save_path = os.path.join(out_dir, record["crop_path"])
                            ok = cv2.imwrite(crop_save_path, crop_resized)
                            if not ok:
                                # First failure: warn. Don't spam for every crop.
                                if crop_idx <= 5 or crop_idx % 500 == 0:
                                    print(f"\n  WARNING: cv2.imwrite failed at crop {crop_idx} "
                                          f"— disk full or quota exceeded?")
                                record["crop_saved"] = False
                            else:
                                record["crop_saved"] = True
                        else:
                            record["crop_saved"] = False
                    else:
                        record["crop_saved"] = False

                    crop_idx += 1

        # ── Incremental save: checkpoint metadata every 100 batches ──────
        if batch_idx > 0 and batch_idx % 100 == 0:
            n_tp_cur = sum(1 for r in all_records if r["is_tp"])
            n_fp_cur = sum(1 for r in all_records if not r["is_tp"])
            n_saved_cur = sum(1 for r in all_records if r.get("crop_saved", False))

            meta_tmp = os.path.join(out_dir, "metadata_checkpoint.json")
            with open(meta_tmp, "w") as f:
                json.dump({
                    "cond": cond_key, "label": label,
                    "n_detections": len(all_records),
                    "n_tp": n_tp_cur, "n_fp": n_fp_cur,
                    "n_crops_saved": n_saved_cur,
                    "conf_threshold": CONF_THRESHOLD,
                    "crop_context_factor": CROP_CONTEXT_FACTOR,
                    "crop_target_size": list(CROP_TARGET_SIZE),
                    "in_progress": True,
                    "batch_idx": batch_idx,
                    "records": all_records,
                }, f, indent=2)

            # Also process IR for metadata (but don't extract IR crops since
            # the primary analysis is on RGB)
            # We skip IR crop extraction for now — it's a secondary analysis.

    # ── Summary & Save ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    n_tp = sum(1 for r in all_records if r["is_tp"])
    n_fp = sum(1 for r in all_records if not r["is_tp"])
    n_crops_saved = sum(1 for r in all_records if r.get("crop_saved", False))

    print(f"\n{'─'*50}")
    print(f"Extraction complete: {len(all_records)} detections")
    print(f"  TP: {n_tp}, FP: {n_fp}")
    print(f"  Crops saved: {n_crops_saved}/{len(all_records)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_records)*1000:.1f} ms/det)")
    print(f"{'─'*50}")

    # Save metadata
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "cond": cond_key,
            "label": label,
            "n_detections": len(all_records),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_crops_saved": n_crops_saved,
            "conf_threshold": CONF_THRESHOLD,
            "crop_context_factor": CROP_CONTEXT_FACTOR,
            "crop_target_size": list(CROP_TARGET_SIZE),
            "records": all_records,
        }, f, indent=2)

    print(f"Metadata saved to: {meta_path}")
    return all_records


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Extract detection crops for E3 balanced-classification positive control"
    )
    parser.add_argument("--cond", type=str, default="d",
                       help="Condition key: d (Average) or e (DS)")
    parser.add_argument("--out-dir", type=str, default=OUTPUT_DIR,
                       help="Output root directory for crops and metadata")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Limit number of images for quick test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.cond not in CONDITIONS:
        print(f"Unknown condition '{args.cond}'. Available: {list(CONDITIONS.keys())}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extract_crops_for_condition(args.cond, CONDITIONS[args.cond], device,
                               args.max_images, out_root=args.out_dir)
    print("\nDone. Next step: run tools/finetune_crops_edl.py")


if __name__ == "__main__":
    main()
