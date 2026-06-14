#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Recompute E3 crop-control CIs with proper sequence-level clustering.
=====================================================================

The original finetune_crops_edl.py::sequence_clustered_bootstrap() resampled by
img_id (individual frames), not by the 121 test sequences. This script fixes that:
  1. Loads the frozen model checkpoint + crop metadata
  2. Runs inference-only evaluation on test crops (no training)
  3. Maps each crop's img_id → sequence name via COCO file_name
  4. Computes sequence-clustered bootstrap CIs (resample 121 sequences)
  5. Reports scale-stratified results

No GPU training needed — inference only. Saves per-crop predictions so CIs can
be recomputed offline.

Usage on Snellius:
    module purge && module load 2023
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
    source /gpfs/work5/0/prjs1970/envs/mm-uav-venv/bin/activate

    cd /gpfs/work5/0/prjs1970/code/MM-UAV-Benchmark
    python tools/recompute_sequence_cis.py --cond d
    python tools/recompute_sequence_cis.py --cond e
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yolox.exp import get_exp
from yolox.models.yolo_head_evidential import softplus_evidence

# ── Config ───────────────────────────────────────────────────────────────────

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

CROP_DIR = os.environ.get("CROP_DIR", "/path/to/e3_crops")
OUT_DIR = os.environ.get("OUT_DIR", "/path/to/e3_crop_control_v2")
BATCH_SIZE = 32
NUM_WORKERS = 2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Sequence mapping ─────────────────────────────────────────────────────────

def build_seq_mapping(coco_api):
    """Build img_id → sequence_name mapping from COCO image metadata.

    MM-UAV file_name format: '<seq_name>/rgb_frame/<frame>.jpg'
    (see coco2.py line 301: "tar indexes use the sequence name only").
    Returns: dict img_id → seq_name
    """
    mapping = {}
    for img_info in coco_api.loadImgs(coco_api.getImgIds()):
        file_name = img_info["file_name"]
        # Sequence name is the first path component
        seq_name = file_name.split("/")[0]
        mapping[img_info["id"]] = seq_name
    return mapping


# ── Dataset (inference-only) ─────────────────────────────────────────────────

class CropEvalDataset(Dataset):
    """Load all saved detection crops for inference-only evaluation."""

    def __init__(self, metadata_path, crop_root):
        with open(metadata_path) as f:
            meta = json.load(f)
        self.records = [r for r in meta["records"] if r.get("crop_saved", False)]
        self.crop_root = crop_root
        self.target_size = (640, 640)
        print(f"CropEvalDataset: {len(self.records)} crops")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        crop_path = os.path.join(self.crop_root, rec["crop_path"])
        import cv2
        img = cv2.imread(crop_path)
        if img is None:
            return torch.zeros(3, *self.target_size), idx
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), idx


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_uauc(uncertainties, is_error):
    """AUROC of uncertainty predicting errors."""
    n = len(uncertainties)
    if n < 2:
        return 0.5
    n_err = is_error.sum()
    n_ok = n - n_err
    if n_err == 0 or n_ok == 0:
        return 0.5
    order = np.argsort(-uncertainties)
    is_err_sorted = is_error[order]
    ranks = np.arange(1, n + 1)
    err_ranks = ranks[is_err_sorted == 1]
    return float((err_ranks.sum() - n_err * (n_err + 1) / 2) / (n_err * n_ok))


def sequence_clustered_bootstrap(seq_ids, vacuity, is_error, n_bootstrap=2000, seed=42):
    """Bootstrap UAUC with proper sequence-level resampling.

    Args:
        seq_ids: np.array of sequence names, one per crop
        vacuity: np.array of vacuity values
        is_error: np.array of bool (True = error / FP)
        n_bootstrap: number of bootstrap resamples
        seed: random seed

    Returns:
        dict with uauc_mean, uauc_std, uauc_ci95_low, uauc_ci95_high, n_seqs
    """
    rng = np.random.RandomState(seed)
    unique_seqs = np.unique(seq_ids)
    n_seqs = len(unique_seqs)

    uaucs = []
    for _ in range(n_bootstrap):
        # Resample sequences with replacement
        sampled = rng.choice(unique_seqs, size=n_seqs, replace=True)
        mask = np.isin(seq_ids, sampled)
        if mask.sum() < 2:
            uaucs.append(0.5)
            continue
        u_sample = vacuity[mask]
        err_sample = is_error[mask]
        valid = ~np.isnan(u_sample)
        if valid.sum() < 2:
            uaucs.append(0.5)
        else:
            uaucs.append(compute_uauc(u_sample[valid], err_sample[valid]))

    uaucs = np.array(uaucs)
    return {
        "uauc_mean": float(np.mean(uaucs)),
        "uauc_std": float(np.std(uaucs)),
        "uauc_ci95_low": float(np.percentile(uaucs, 2.5)),
        "uauc_ci95_high": float(np.percentile(uaucs, 97.5)),
        "n_sequences": n_seqs,
    }


# ── Scale stratification ─────────────────────────────────────────────────────

COCO_SCALE_BINS = {
    "small": (0, 32**2),        # < 1024 px²
    "medium": (32**2, 96**2),   # 1024–9216 px²
    "large": (96**2, 1e9),      # > 9216 px²
}


def get_bbox_area(rec):
    """Compute original bbox area in px²."""
    b = rec["bbox_xyxy"]
    return (b[2] - b[0]) * (b[3] - b[1])


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_compute_cis(cond_key, cond_info, device, args):
    """Full evaluation pipeline: load model → run inference → compute CIs."""
    label = cond_info["label"]
    exp_file = os.path.join(ROOT, cond_info["exp_file"])
    ckpt_path = os.path.join(ROOT, cond_info["ckpt"])
    meta_path = os.path.join(args.crop_dir, cond_key, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found.")
        return None

    out_dir = os.path.join(args.out_dir, cond_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Sequence-clustered CI recomputation: ({cond_key}) {label}")
    print(f"  Crop metadata: {meta_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # ── Load model ────────────────────────────────────────────────────────
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.eval()
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"  Model loaded.")

    # ── Create dataset and dataloader ─────────────────────────────────────
    crop_root = os.path.join(args.crop_dir, cond_key)
    ds = CropEvalDataset(meta_path, crop_root)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=False)

    # ── Build sequence mapping from eval COCO ──────────────────────────────
    # Use the experiment's eval data loader to access the COCO dataset,
    # which maps img_id → file_name (containing the sequence name).
    print("  Building sequence mapping from eval data loader...")
    val_loader1, val_loader2 = exp.get_eval_loader(
        batch_size=1, is_distributed=False
    )
    coco = val_loader1.dataset.coco
    seq_mapping = build_seq_mapping(coco)
    print(f"  Sequence mapping: {len(seq_mapping)} images → "
          f"{len(set(seq_mapping.values()))} sequences")

    # ── Run inference ─────────────────────────────────────────────────────
    all_preds = []
    all_records = ds.records
    seq_id_by_idx = []  # sequence name for each crop

    print(f"  Running inference on {len(ds)} crops...")
    for imgs, indices in tqdm(loader, desc="Inference"):
        imgs = imgs.to(device)
        B = imgs.shape[0]

        # Pass same crop to both streams (matches original DS-arm design)
        out1, out2 = model(imgs, imgs)

        cls_logits = model.head._last_cls_logits
        evidence = softplus_evidence(cls_logits)
        alphas = evidence + 1.0
        S_eff = alphas.sum(dim=-1, keepdim=True) + 1.0
        p_uav = alphas.squeeze(-1) / S_eff.squeeze(-1)
        u = 1.0 / S_eff.squeeze(-1)

        max_vals, max_indices = p_uav.max(dim=1)

        for i in range(B):
            idx = indices[i].item()
            rec = all_records[idx]
            img_id = rec["img_id"]

            # Map img_id to sequence
            seq_name = seq_mapping.get(img_id, f"img_{img_id}")
            seq_id_by_idx.append(seq_name)

            all_preds.append({
                "img_id": img_id,
                "seq_name": seq_name,
                "crop_id": rec["crop_id"],
                "p_uav": float(p_uav[i, max_indices[i]].item()),
                "vacuity": float(u[i, max_indices[i]].item()),
                "is_error": not rec["is_tp"],  # FP → error
                "bbox_area": get_bbox_area(rec),
            })

    # ── Save per-crop predictions ─────────────────────────────────────────
    preds_path = os.path.join(out_dir, "per_crop_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(all_preds, f, indent=2)
    print(f"  Saved {len(all_preds)} per-crop predictions to {preds_path}")

    # ── Compute aggregate sequence-clustered CI ───────────────────────────
    seq_ids_all = np.array([p["seq_name"] for p in all_preds])
    vacuity_all = np.array([p["vacuity"] for p in all_preds])
    is_error_all = np.array([p["is_error"] for p in all_preds])
    n_seqs_unique = len(np.unique(seq_ids_all))

    print(f"\n  Aggregate: {len(all_preds)} crops, {n_seqs_unique} sequences")

    agg_result = sequence_clustered_bootstrap(
        seq_ids_all, vacuity_all, is_error_all,
        n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    print(f"  Vacuity UAUC: {agg_result['uauc_mean']:.4f} "
          f"[{agg_result['uauc_ci95_low']:.4f}, {agg_result['uauc_ci95_high']:.4f}]")
    print(f"  Criterion (CI excludes 0.5): "
          f"{'MET' if agg_result['uauc_ci95_low'] > 0.5 else 'NOT MET'}")

    # ── Compute scale-stratified CIs ──────────────────────────────────────
    scale_results = {}
    for bin_name, (lo, hi) in COCO_SCALE_BINS.items():
        bin_mask = np.array([lo <= p["bbox_area"] < hi for p in all_preds])
        n_bin = bin_mask.sum()
        if n_bin < 10:
            scale_results[bin_name] = None
            continue
        bin_seqs = seq_ids_all[bin_mask]
        bin_vac = vacuity_all[bin_mask]
        bin_err = is_error_all[bin_mask]
        n_seqs_bin = len(np.unique(bin_seqs))

        bin_result = sequence_clustered_bootstrap(
            bin_seqs, bin_vac, bin_err,
            n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        bin_result["n_crops"] = int(n_bin)
        scale_results[bin_name] = bin_result
        print(f"  Scale {bin_name:>8s}: UAUC={bin_result['uauc_mean']:.4f} "
              f"[{bin_result['uauc_ci95_low']:.4f}, {bin_result['uauc_ci95_high']:.4f}] "
              f"(n={n_bin} crops, {n_seqs_bin} seqs)")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "cond": cond_key,
        "label": label,
        "bootstrap_type": "sequence-clustered (121 sequences resampled)",
        "n_bootstrap": args.n_bootstrap,
        "n_total_crops": len(all_preds),
        "n_sequences": n_seqs_unique,
        "aggregate_uauc": agg_result,
        "scale_stratified_uauc": scale_results,
        "success_criterion_met": agg_result["uauc_ci95_low"] > 0.5,
        "success_summary": (
            f"UAUC = {agg_result['uauc_mean']:.4f} "
            f"[{agg_result['uauc_ci95_low']:.4f}, {agg_result['uauc_ci95_high']:.4f}]. "
            f"CI {'excludes' if agg_result['uauc_ci95_low'] > 0.5 else 'includes'} 0.5. "
            f"Criterion: {'MET' if agg_result['uauc_ci95_low'] > 0.5 else 'NOT MET'}."
        ),
    }

    results_path = os.path.join(out_dir, "sequence_clustered_cis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"SUCCESS CRITERION: {results['success_summary']}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Recompute E3 crop-control CIs with sequence-level clustering"
    )
    parser.add_argument("--cond", type=str, default="d",
                       help="Condition: d (Average) or e (DS)")
    parser.add_argument("--crop-dir", type=str, default=CROP_DIR,
                       help="Root directory with crop metadata from Step 1")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR,
                       help="Output directory for recomputed results")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                       help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.cond not in CONDITIONS:
        print(f"Unknown condition '{args.cond}'. Available: {list(CONDITIONS.keys())}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    evaluate_and_compute_cis(args.cond, CONDITIONS[args.cond], device, args)
    print("\nDone. Rsync results back and update paper numbers.")


if __name__ == "__main__":
    main()
