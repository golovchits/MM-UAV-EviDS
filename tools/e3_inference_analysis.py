#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EviDS-UAV E3: Inference Analysis Script
#
# Runs each E3 checkpoint on the MM-UAV validation set and computes:
#   1. Detection-level ECE (post-NMS, IoU-matched to GT)
#   2. UAUC (1-confidence) — detection-level, GT-based
#   3. Per-anchor uncertainty histograms (u = K/S distribution)
#   4. Agree-vs-disagree: stratify by inter-modal disagreement D = |p_rgb - p_ir|
#
# Notes:
#   - For K=1 with alpha_bg=1, DS conflict C is structurally zero.
#     We use D = |p_rgb - p_ir| as the inter-modal disagreement metric.
#   - ECE is computed on NMS-filtered detections (IoU≥0.5 to GT = correct).
#   - UAUC is detection-level only (GT-matched); vacuity UAUC requires
#     per-anchor GT assignment which is not available from standard eval.
#
# Usage:
#   python tools/e3_inference_analysis.py --cond c
#   python tools/e3_inference_analysis.py --all
#
# Outputs: tools/e3_analysis/e3_analysis_{cond}.json

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.exp import get_exp
from yolox.utils import postprocess


# ── Helpers ──────────────────────────────────────────────────────────────────

def evidential_stats(cls_logits, num_classes=1):
    """Compute Dirichlet probabilities, vacuity, entropy from logits [N, K]."""
    evidence = F.softplus(cls_logits)
    alphas = evidence + 1.0
    S = alphas.sum(dim=-1, keepdim=True)
    # Effective Dirichlet: K_eff = num_classes + 1 (explicit bg α=1),
    # S_eff = S + 1.0.  Vacuity u = K_eff / S_eff.
    u = (num_classes + 1) / (S + 1.0 + 1e-8)
    p = alphas / (S + 1.0 + 1e-8)  # ∈ [0.5, 1) for K=1
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    return {"p": p.squeeze(-1), "u": u.squeeze(-1), "entropy": entropy}


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def compute_ece(confidences, accuracies, n_bins=15):
    if len(confidences) == 0:
        return 0.0, []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(confidences, bins[1:-1])
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (n / len(confidences)) * abs(bin_acc - bin_conf)
        bin_stats.append({
            "bin": b, "center": float((bins[b] + bins[b+1]) / 2),
            "accuracy": float(bin_acc), "confidence": float(bin_conf), "count": int(n),
        })
    return ece, bin_stats


def compute_uauc(uncertainties, is_error):
    if len(uncertainties) < 2:
        return 0.5
    order = np.argsort(-uncertainties)
    is_err_sorted = is_error[order]
    n_err = is_err_sorted.sum()
    n_ok = len(is_err_sorted) - n_err
    if n_err == 0 or n_ok == 0:
        return 0.5
    ranks = np.arange(1, len(is_err_sorted) + 1)
    err_ranks = ranks[is_err_sorted == 1]
    return float((err_ranks.sum() - n_err * (n_err + 1) / 2) / (n_err * n_ok))


# ── Main analysis ────────────────────────────────────────────────────────────

def _match_det_vacuity(decoded_out_b, nms_dets_b, u_per_anchor):
    """Return per-detection vacuity by matching each post-NMS box to its pre-NMS anchor.

    decoded_out_b: [N, 5+K] model decoded output for one image (cx,cy,w,h,...) in input space
    nms_dets_b: [D, 7] post-NMS detections (x1,y1,x2,y2,...) in input space, or None
    u_per_anchor: [N] vacuity tensor (CPU)
    """
    if nms_dets_b is None or len(nms_dets_b) == 0:
        return np.array([], dtype=np.float32)
    pre = decoded_out_b[:, :4].cpu()
    pre_corners = torch.stack([
        pre[:, 0] - pre[:, 2] / 2, pre[:, 1] - pre[:, 3] / 2,
        pre[:, 0] + pre[:, 2] / 2, pre[:, 1] + pre[:, 3] / 2,
    ], dim=1)
    det_boxes = nms_dets_b[:, :4].cpu()
    ious = box_iou(det_boxes, pre_corners)  # [D, N]
    best_idx = ious.argmax(dim=1)           # [D]
    return u_per_anchor[best_idx].numpy()


def analyze_condition(exp_file, ckpt_path, cond_label, device="cuda", max_images=None,
                      temp_scale_T=None):
    print(f"\n{'='*60}")
    print(f"Analyzing: {cond_label}")
    print(f"  Exp: {exp_file}")
    print(f"  Ckpt: {ckpt_path}")
    if temp_scale_T is not None:
        print(f"  Temperature scaling: T={temp_scale_T:.4f}")

    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"  Model loaded (best epoch ~{ckpt.get('start_epoch', '?')})")

    if temp_scale_T is not None:
        if hasattr(model, 'head') and hasattr(model.head, 'T'):
            model.head.T = temp_scale_T
        if hasattr(model, 'head2') and hasattr(model.head2, 'T'):
            model.head2.T = temp_scale_T
        print(f"  Applied T={temp_scale_T:.4f} to both heads.")

    val_loader1, val_loader2 = exp.get_eval_loader(batch_size=8, is_distributed=False)
    num_classes = exp.num_classes
    is_evidential = "evidential" in exp_file.lower()
    is_ds_model = hasattr(model, 'ds_fusion') and model.ds_fusion is not None

    # Detection-level collections (post-NMS)
    det_scores = {"rgb": [], "ir": []}
    det_correct = {"rgb": [], "ir": []}

    # Per-anchor collections (evidential only)
    anchor_u = {"rgb": [], "ir": []}
    anchor_ent = {"rgb": [], "ir": []}
    anchor_p = {"rgb": [], "ir": []}

    # Per-detection vacuity (evidential only) — matched from pre-NMS anchor via box IoU
    det_u = {"rgb": [], "ir": []}

    # Agree-vs-disagree (DS only): per-anchor fused uncertainty u_fused
    anchor_u_fused = []

    n_images = 0
    for (imgs1, _, infos1, ids1), (imgs2, _, infos2, ids2) in zip(
        tqdm(val_loader1, desc="RGB+IR"), val_loader2
    ):
        if max_images and n_images >= max_images:
            break

        imgs1, imgs2 = imgs1.to(device), imgs2.to(device)

        with torch.no_grad():
            outputs1, outputs2 = model(imgs1, imgs2)

        # Capture per-head logits BEFORE postprocess (which modifies outputs)
        cls_logits1 = getattr(model.head, '_last_cls_logits', None) if is_evidential else None
        cls_logits2 = getattr(model.head2, '_last_cls_logits', None) if is_evidential else None
        u_fused_batch = getattr(model, '_last_u_fused', None)

        # Fast batch NMS via built-in postprocess
        dets1 = postprocess(outputs1.clone(), num_classes, conf_thre=0.001, nms_thre=0.65)
        dets2 = postprocess(outputs2.clone(), num_classes, conf_thre=0.001, nms_thre=0.65)

        for b in range(imgs1.shape[0]):
            n_images += 1

            img_id1 = int(ids1[b].item()) if torch.is_tensor(ids1[b]) else int(ids1[b])
            img_id2 = int(ids2[b].item()) if torch.is_tensor(ids2[b]) else int(ids2[b])
            img_h1, img_w1 = int(infos1[0][b].item()), int(infos1[1][b].item())
            img_h2, img_w2 = int(infos2[0][b].item()), int(infos2[1][b].item())

            for stream, dets, img_id, img_h, img_w, loader in [
                ("rgb", dets1, img_id1, img_h1, img_w1, val_loader1),
                ("ir", dets2, img_id2, img_h2, img_w2, val_loader2),
            ]:
                ann_ids = loader.dataset.coco.getAnnIds(imgIds=img_id)
                anns = loader.dataset.coco.loadAnns(ann_ids)
                gt_boxes = torch.tensor(
                    [[a['bbox'][0], a['bbox'][1],
                      a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                     for a in anns], dtype=torch.float32
                ) if anns else torch.zeros((0, 4))

                pred = dets[b]
                if pred is None or pred.shape[0] == 0:
                    det_scores[stream].append(np.array([]))
                    det_correct[stream].append(np.array([]))
                    continue

                # postprocess output: [x1, y1, x2, y2, obj_conf, cls_conf, cls_idx]
                boxes = pred[:, :4].cpu()
                scale = min(exp.test_size[0] / img_h, exp.test_size[1] / img_w)
                boxes /= scale
                scores = (pred[:, 4] * pred[:, 5]).cpu()  # obj_conf * cls_conf

                # IoU matching (already NMS-filtered, just match to GT)
                if len(gt_boxes) == 0:
                    det_scores[stream].append(scores.numpy())
                    det_correct[stream].append(np.zeros(len(scores), dtype=bool))
                else:
                    ious = box_iou(boxes, gt_boxes)
                    max_ious, _ = ious.max(dim=1)
                    is_correct = (max_ious >= 0.5).numpy()
                    det_scores[stream].append(scores.numpy())
                    det_correct[stream].append(is_correct)

            # Per-anchor evidential stats
            if cls_logits1 is not None:
                s1 = evidential_stats(cls_logits1[b].cpu(), num_classes)
                anchor_u["rgb"].append(s1["u"].numpy())
                anchor_ent["rgb"].append(s1["entropy"].numpy())
                anchor_p["rgb"].append(s1["p"].numpy())
                det_u["rgb"].append(_match_det_vacuity(outputs1[b], dets1[b], s1["u"]))
            if cls_logits2 is not None:
                s2 = evidential_stats(cls_logits2[b].cpu(), num_classes)
                anchor_u["ir"].append(s2["u"].numpy())
                anchor_ent["ir"].append(s2["entropy"].numpy())
                anchor_p["ir"].append(s2["p"].numpy())
                det_u["ir"].append(_match_det_vacuity(outputs2[b], dets2[b], s2["u"]))

            # Agree-vs-disagree: capture DS fused uncertainty
            if is_ds_model and u_fused_batch is not None:
                anchor_u_fused.append(u_fused_batch[b].squeeze(-1).cpu().numpy())

    # ── Aggregate ──
    results = {"condition": cond_label, "n_images": n_images}

    for stream in ["rgb", "ir"]:
        scores = np.concatenate(det_scores[stream]) if det_scores[stream] else np.array([])
        correct = np.concatenate(det_correct[stream]) if det_correct[stream] else np.array([])

        if len(scores) > 0:
            ece_val, ece_bins = compute_ece(scores, correct)
            results[f"{stream}_ece"] = ece_val
            results[f"{stream}_ece_bins"] = ece_bins
            results[f"{stream}_n_dets"] = int(len(scores))
            results[f"{stream}_precision"] = float(correct.mean())

            is_error = 1.0 - correct.astype(np.float32)
            results[f"{stream}_uauc_1mconf"] = compute_uauc(1.0 - scores, is_error)

            # UAUC_vac: detection-level vacuity ranks FP above TP (evidential only)
            if det_u[stream]:
                u_dets = np.concatenate(det_u[stream])
                if len(u_dets) == len(correct) and len(u_dets) > 1:
                    results[f"{stream}_uauc_vac"] = compute_uauc(u_dets, is_error)
                    print(f"  {stream}: UAUC_vac={results[f'{stream}_uauc_vac']:.4f}, "
                          f"u_det mean={u_dets.mean():.6f} std={u_dets.std():.6f}")
        else:
            results[f"{stream}_ece"] = 0.0
            results[f"{stream}_n_dets"] = 0
            results[f"{stream}_uauc_1mconf"] = 0.5

        # Per-anchor uncertainty stats (evidential only, no UAUC — vacuity UAUC
        # requires GT-assigned per-anchor correctness which is not available)
        if anchor_u[stream]:
            u_all = np.concatenate(anchor_u[stream])
            results[f"{stream}_u_mean"] = float(u_all.mean())
            results[f"{stream}_u_std"] = float(u_all.std())
            results[f"{stream}_u_median"] = float(np.median(u_all))
            results[f"{stream}_u_pct_below_01"] = float((u_all < 0.1).mean())
            results[f"{stream}_u_hist"] = np.histogram(u_all, bins=50, range=(0, 1))[0].tolist()
            results[f"{stream}_u_hist_edges"] = np.histogram(u_all, bins=50, range=(0, 1))[1].tolist()

            print(f"  {stream}: ECE={ece_val:.4f}, n_dets={len(scores)}, "
                  f"u_mean={results[f'{stream}_u_mean']:.4f}, "
                  f"u_median={results[f'{stream}_u_median']:.4f}, "
                  f"u_pct<0.1={results[f'{stream}_u_pct_below_01']:.3f}")

    # ── Agree-vs-disagree (DS condition only) ──
    # Compute D = |p_rgb - p_ir| from already-collected per-anchor p values.
    # For K=1 with alpha_bg=1, DS conflict C is structurally 0, so we use
    # per-head probability disagreement as the inter-modal conflict metric.
    if is_ds_model and anchor_p["rgb"] and anchor_p["ir"]:
        p_rgb_all = np.concatenate(anchor_p["rgb"])
        p_ir_all = np.concatenate(anchor_p["ir"])
        D_all = np.abs(p_rgb_all - p_ir_all)

        results["disagreement_D_mean"] = float(D_all.mean())
        results["disagreement_D_std"] = float(D_all.std())
        results["disagreement_D_median"] = float(np.median(D_all))

        if anchor_u_fused:
            uf_all = np.concatenate(anchor_u_fused)
            n_common = min(len(D_all), len(uf_all))
            D_all = D_all[:n_common]
            uf_all = uf_all[:n_common]

            # Correlation: D vs u_fused
            if n_common > 2 and D_all.std() > 1e-12 and uf_all.std() > 1e-12:
                corr = np.corrcoef(D_all, uf_all)[0, 1]
                results["disagreement_D_vs_u_fused_pearson"] = float(corr) if not np.isnan(corr) else None
            else:
                results["disagreement_D_vs_u_fused_pearson"] = None

            # Stratify by D tertiles
            d_lo = np.percentile(D_all, 33)
            d_hi = np.percentile(D_all, 67)
            strata_def = {
                "low": D_all < d_lo,
                "medium": (D_all >= d_lo) & (D_all < d_hi),
                "high": D_all >= d_hi,
            }
            results["disagreement_strata"] = {}
            for name, mask in strata_def.items():
                n = mask.sum()
                if n == 0:
                    continue
                results["disagreement_strata"][name] = {
                    "n": int(n),
                    "frac": float(n / len(D_all)),
                    "D_range": [float(D_all[mask].min()), float(D_all[mask].max())],
                    "u_fused_mean": float(uf_all[mask].mean()),
                    "u_fused_std": float(uf_all[mask].std()),
                }

            print(f"  Agree-vs-disagree: D_mean={D_all.mean():.4f}, "
                  f"corr(D, u_fused)={results.get('disagreement_D_vs_u_fused_pearson', 'N/A')}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("E3 Inference Analysis")
    parser.add_argument("--cond", type=str, default=None, help="Single condition: a,b,c,d,e")
    parser.add_argument("--all", action="store_true", help="Run all conditions")
    parser.add_argument("--output-dir", type=str, default="tools/e3_analysis")
    parser.add_argument("--max-images", type=int, default=None, help="Limit for quick test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    CONDITIONS = {
        "a": ("MMA-SORT DefConv",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/best_ckpt.pth.tar"),
              None),
        "b": ("MMA-SORT STN",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_stn-tuning-fusion-head.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_stn-tuning-fusion-head/best_ckpt.pth.tar"),
              None),
        "c": ("Evidential + ADFM",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_evidential_adfm.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_evidential_adfm/best_ckpt.pth.tar"),
              None),
        "d": ("Evidential + Average",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_evidential_average.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_evidential_average/best_ckpt.pth.tar"),
              None),
        "e": ("Evidential + DS",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_evidential_ds.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_evidential_ds/best_ckpt.pth.tar"),
              None),
        "g": ("DefConv + TempScale",
              os.path.join(BASE, "yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/best_ckpt.pth.tar"),
              os.path.join(BASE, "YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json")),
    }

    conds = list(CONDITIONS.keys()) if args.all else ([args.cond] if args.cond else [])
    if not conds:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for key in conds:
        label, exp_file, ckpt, temp_json = CONDITIONS[key]
        temp_scale_T = None
        if temp_json is not None:
            if not os.path.exists(temp_json):
                print(f"ERROR: temperature.json not found at {temp_json}")
                print("  Run fit_temperature.py first to generate it.")
                sys.exit(1)
            with open(temp_json) as f:
                temp_data = json.load(f)
            temp_scale_T = float(temp_data["T"])
            print(f"  Loaded T={temp_scale_T:.4f} from {temp_json}")
        results = analyze_condition(exp_file, ckpt, f"({key}) {label}",
                                    device=args.device, max_images=args.max_images,
                                    temp_scale_T=temp_scale_T)
        all_results[key] = results

        out_path = os.path.join(args.output_dir, f"e3_analysis_{key}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        print(f"  Saved: {out_path}")

    summary_path = os.path.join(args.output_dir, "e3_analysis_summary.json")
    # Merge with existing summary so a single-condition run doesn't erase others.
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass
    existing.update(all_results)
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    # Quick summary
    print("\n" + "=" * 100)
    print("E3 Analysis Summary")
    print("=" * 100)
    header = f"{'Condition':<28} {'ECE_RGB':>8} {'ECE_IR':>8} {'u_pct<0.1_RGB':>13} {'u_pct<0.1_IR':>13} {'u_mean_RGB':>10} {'u_mean_IR':>10}"
    print(header)
    print("-" * 100)
    for key, r in all_results.items():
        print(f"{r['condition']:<28} {r.get('rgb_ece',0):8.4f} {r.get('ir_ece',0):8.4f} "
              f"{r.get('rgb_u_pct_below_01',0):13.4f} {r.get('ir_u_pct_below_01',0):13.4f} "
              f"{r.get('rgb_u_mean',0):10.4f} {r.get('ir_u_mean',0):10.4f}")

    if "a" in all_results and "g" in all_results:
        ece_a = all_results["a"].get("rgb_ece", 0)
        ece_g = all_results["g"].get("rgb_ece", 0)
        direction = "improved" if ece_g < ece_a else "WORSENED"
        print(f"\n  TempScale (g) vs baseline (a): ECE_RGB {ece_a:.4f} -> {ece_g:.4f} ({direction})")

    if "c" in all_results and "e" in all_results:
        uc = all_results["c"].get("rgb_u_pct_below_01", 0)
        ue = all_results["e"].get("rgb_u_pct_below_01", 0)
        print(f"\n  Key: ADFM fraction(u<0.1)={uc:.3f} vs DS fraction(u<0.1)={ue:.3f}")
        if uc > ue:
            print(f"  ADFM has MORE anchors with collapsed uncertainty (u<0.1) — "
                  f"validates cls_loss narrative (ADFM ~0.002 vs DS ~0.3)")
        r = all_results["e"]
        if "disagreement_D_mean" in r:
            print(f"  DS agree-vs-disagree: D_mean={r['disagreement_D_mean']:.4f}, "
                  f"corr(D,u_fused)={r.get('disagreement_D_vs_u_fused_pearson', 'N/A')}")


if __name__ == "__main__":
    main()
