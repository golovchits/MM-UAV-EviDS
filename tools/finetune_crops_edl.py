#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
E3 Balanced-Crop Positive Control — Fine-Tuning (Path B, redesigned)
=====================================================================

Instead of inference-only evaluation (which can't change UAUC since AUROC is
prevalence-invariant), this script FINE-TUNES the evidential classification
head on a balanced set of detection crops, then measures vacuity UAUC on a
held-out test set. This mirrors E2's balanced classification regime inside
E3's data and architecture — the causal isolation the thesis needs.

Design (per writer-agent feedback):
  1. Extract crops at E3 detection locations (Step 1 — already done)
  2. Split crops by SEQUENCE into train/val/test to prevent leakage
  3. Freeze backbone + FPN + OGAA + regression/objectness branches
  4. Fine-tune ONLY the evidential classification branch (cls_convs + cls_preds)
     on a balanced (1:1 TP:FP) training set using EDL loss
  5. Measure vacuity UAUC on held-out test crops
  6. Report scale-stratified UAUC (to rule out scale confound)
  7. Pre-registered success criterion: UAUC > 0.5 with sequence-clustered
     bootstrap CI excluding 0.5

Expected GPU time: ~15-30 min on A100 for ~12K crops, 10 epochs.

Usage on Snellius (interactive or batch):
    module purge && module load 2023
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
    source /gpfs/work5/0/prjs1970/envs/mm-uav-venv/bin/activate

    cd /gpfs/work5/0/prjs1970/code/MM-UAV-Benchmark
    python tools/finetune_crops_edl.py --cond d --epochs 10 --lr 1e-3
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yolox.exp import get_exp
from yolox.models.yolo_head_evidential import softplus_evidence, edl_loss


# ── Configuration ──────────────────────────────────────────────────────────

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

CROP_DIR_DEFAULT = os.environ.get("CROP_DIR", "/path/to/e3_crops")
OUT_DIR_DEFAULT = os.environ.get("OUT_DIR", "/path/to/e3_finetune")
BATCH_SIZE = 32
NUM_WORKERS = 2

# ImageNet normalization (same as YOLOX)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────

class CropClassificationDataset(Dataset):
    """Dataset over extracted detection crops with sequence-aware splitting.

    Each crop is labeled 1 (UAV / original TP) or 0 (background / original FP).
    Split is by SEQUENCE (img_id groups) to prevent leakage across train/val/test.
    """

    def __init__(self, metadata_path, crop_root, split='train',
                 train_seqs=None, val_seqs=None, test_seqs=None,
                 balance=True, seed=42):
        with open(metadata_path) as f:
            meta = json.load(f)

        self.crop_root = crop_root

        # Filter to crops that were successfully saved
        records = [r for r in meta["records"] if r.get("crop_saved", False)]

        # Assign sequences deterministically
        all_seqs = sorted(set(r["img_id"] for r in records))
        rng = np.random.RandomState(seed)
        rng.shuffle(all_seqs)
        n = len(all_seqs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        if train_seqs is None:
            train_seqs = set(all_seqs[:n_train])
            val_seqs = set(all_seqs[n_train:n_train + n_val])
            test_seqs = set(all_seqs[n_train + n_val:])

        self.train_seqs = train_seqs
        self.val_seqs = val_seqs
        self.test_seqs = test_seqs

        if split == 'train':
            seqs = train_seqs
        elif split == 'val':
            seqs = val_seqs
        else:
            seqs = test_seqs

        self.records = [r for r in records if r["img_id"] in seqs]

        if balance and split == 'train':
            # Balance TP:FP to 1:1 for training
            tps = [r for r in self.records if r["is_tp"]]
            fps = [r for r in self.records if not r["is_tp"]]
            n_min = min(len(tps), len(fps))
            if n_min > 0:
                rng_train = np.random.RandomState(seed + 1)
                tps = rng_train.choice(tps, size=n_min, replace=False).tolist()
                fps = rng_train.choice(fps, size=n_min, replace=False).tolist()
                self.records = tps + fps
                rng_train.shuffle(self.records)

        self.split = split
        self.target_size = (640, 640)
        print(f"CropDataset [{split}]: {len(self.records)} crops "
              f"(TP={sum(1 for r in self.records if r['is_tp'])}, "
              f"FP={sum(1 for r in self.records if not r['is_tp'])})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        crop_path = os.path.join(self.crop_root, rec["crop_path"])
        img = cv2.imread(crop_path)
        if img is None:
            return torch.zeros(3, *self.target_size), torch.tensor(0.0), idx

        # BGR → RGB, [0, 255] → [0, 1]
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)  # HWC → CHW

        label = 1.0 if rec["is_tp"] else 0.0
        return torch.from_numpy(img).float(), torch.tensor(label).float(), idx


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_uauc(uncertainties, is_error):
    """AUROC of uncertainty predicting errors. > 0.5 = correct ranking."""
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


def sequence_clustered_bootstrap(records, preds, n_bootstrap=1000, seed=42):
    """Bootstrap UAUC with sequence-clustered resampling.

    Resamples entire sequences (img_ids) rather than individual crops to
    preserve within-sequence correlation structure.
    """
    rng = np.random.RandomState(seed)
    # Group by img_id
    seq_ids = np.array([r["img_id"] for r in records])
    unique_seqs = np.unique(seq_ids)

    uaucs = []
    for _ in range(n_bootstrap):
        sampled_seqs = rng.choice(unique_seqs, size=len(unique_seqs), replace=True)
        mask = np.isin(seq_ids, sampled_seqs)
        if mask.sum() < 2:
            uaucs.append(0.5)
            continue

        u_sample = np.array([preds[i]["vacuity"] for i in range(len(preds)) if mask[i]])
        err_sample = np.array([preds[i]["is_error"] for i in range(len(preds)) if mask[i]])
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
    }


# ── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, kl_anneal_epochs=10):
    """Fine-tune the evidential classification head on balanced crops."""
    model.eval()  # eval mode to avoid YOLOX training loss path
    # But we need gradients through cls_convs and cls_preds
    # We manually set requires_grad for the classification branches
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for imgs, labels, indices in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)  # [B], 0 or 1
        B = imgs.shape[0]

        # Forward pass: model expects (rgb, ir). Use same crop for both.
        with torch.set_grad_enabled(True):
            # Run in eval mode — we capture cls_logits from head._last_cls_logits
            out1, out2 = model(imgs, imgs)

        # Get per-anchor cls_logits from RGB head
        cls_logits = model.head._last_cls_logits  # [B, N, K]

        # Compute evidential statistics for all anchors
        evidence = softplus_evidence(cls_logits)    # [B, N, K]
        alphas = evidence + 1.0
        S_eff = alphas.sum(dim=-1, keepdim=True) + 1.0
        p_uav = alphas.squeeze(-1) / S_eff.squeeze(-1)  # [B, N]

        # For each crop, take the anchor with highest p_uav
        max_vals, max_indices = p_uav.max(dim=1)  # [B]

        # Gather the cls_logit at the best anchor for each crop
        best_logits = cls_logits[torch.arange(B, device=device), max_indices]  # [B, K]

        # Build binary targets: [B, K] where K=1 for single-class
        # EDL loss expects targets in [B, K] format
        targets = labels.unsqueeze(-1)  # [B, 1]

        # KL annealing coefficient
        kl_lambda = min(1.0, epoch / kl_anneal_epochs)

        # Compute EDL loss on best-anchor logits
        # edl_loss expects alphas, targets, kl_lambda, num_classes
        best_alphas = F.softplus(best_logits) + 1.0  # [B, K]
        loss = edl_loss(best_alphas, targets, kl_lambda, num_classes=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate vacuity UAUC on validation/test set."""
    model.eval()
    all_preds = []

    for imgs, labels, indices in tqdm(loader, desc="Eval"):
        imgs = imgs.to(device)
        B = imgs.shape[0]

        out1, out2 = model(imgs, imgs)

        cls_logits = model.head._last_cls_logits  # [B, N, K]
        evidence = softplus_evidence(cls_logits)
        alphas = evidence + 1.0
        S_eff = alphas.sum(dim=-1, keepdim=True) + 1.0
        p_uav = alphas.squeeze(-1) / S_eff.squeeze(-1)
        u = 2.0 / S_eff.squeeze(-1)  # K_eff=2 (drone + implicit bg α=1)

        max_vals, max_indices = p_uav.max(dim=1)

        for i in range(B):
            all_preds.append({
                "p_uav": float(p_uav[i, max_indices[i]].item()),
                "vacuity": float(u[i, max_indices[i]].item()),
                "is_error": float(labels[i].item()) < 0.5,  # label=0 (FP) → error
                # Actually: label=1 (TP) → not error, label=0 (FP) → error
            })

    return all_preds


# ── Main ───────────────────────────────────────────────────────────────────

def finetune_and_evaluate(cond_key, cond_info, device, args):
    """Full fine-tuning pipeline for one condition."""
    label = cond_info["label"]
    exp_file = os.path.join(ROOT, cond_info["exp_file"])
    ckpt_path = os.path.join(ROOT, cond_info["ckpt"])
    meta_path = os.path.join(args.crop_dir, cond_key, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Run extract_detection_crops.py first.")
        return None

    out_dir = os.path.join(args.out_dir, cond_key)
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already completed (resume on rerun)
    results_path = os.path.join(out_dir, "finetune_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        if prev.get("success_criterion_met") is not None:
            print(f"\nCondition ({cond_key}) already complete — skipping.")
            print(f"  {prev.get('success_summary', '')}")
            return prev

    print(f"\n{'='*70}")
    print(f"Fine-Tuning: ({cond_key}) {label}")
    print(f"  Crop metadata: {meta_path}")
    print(f"  Output: {out_dir}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"{'='*70}")

    # ── Load model ──────────────────────────────────────────────────────
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"  Model loaded.")

    # ── Freeze everything except classification branch ──────────────────
    trainable_params = []
    for name, param in model.named_parameters():
        if 'cls_conv' in name or 'cls_pred' in name:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.1f}%)")
    print(f"  Trainable layers: {len(trainable_params)}")

    # ── Create datasets ─────────────────────────────────────────────────
    crop_root = os.path.join(args.crop_dir, cond_key)
    train_ds = CropClassificationDataset(meta_path, crop_root, split='train',
                                         balance=True, seed=args.seed)
    val_ds = CropClassificationDataset(meta_path, crop_root, split='val',
                                       train_seqs=train_ds.train_seqs,
                                       val_seqs=train_ds.val_seqs,
                                       test_seqs=train_ds.test_seqs,
                                       balance=False, seed=args.seed)
    test_ds = CropClassificationDataset(meta_path, crop_root, split='test',
                                        train_seqs=train_ds.train_seqs,
                                        val_seqs=train_ds.val_seqs,
                                        test_seqs=train_ds.test_seqs,
                                        balance=False, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    # ── Pre-fine-tuning evaluation ──────────────────────────────────────
    print(f"\nPre-fine-tuning evaluation (test set)...")
    preds_before = evaluate(model, test_loader, device)

    # Save sequence info for clustered bootstrap
    test_records = test_ds.records
    uauc_before = sequence_clustered_bootstrap(test_records, preds_before,
                                                n_bootstrap=args.n_bootstrap,
                                                seed=args.seed)
    print(f"  Pre-FT  UAUC: {uauc_before['uauc_mean']:.4f} "
          f"[{uauc_before['uauc_ci95_low']:.4f}, {uauc_before['uauc_ci95_high']:.4f}]")

    # ── Fine-tune ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val_uauc = 0.0
    best_state = None
    history = []
    start_epoch = 1

    # Resume from checkpoint if available
    ckpt_path_ft = os.path.join(out_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path_ft):
        ft_ckpt = torch.load(ckpt_path_ft, map_location=device)
        # Load trainable parameters
        model_state = model.state_dict()
        for k, v in ft_ckpt["model_state"].items():
            if k in model_state:
                model_state[k] = v.to(device)
        model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(ft_ckpt["optimizer_state"])
        start_epoch = ft_ckpt["epoch"] + 1
        best_val_uauc = ft_ckpt.get("best_val_uauc", 0.0)
        history = ft_ckpt.get("history", [])
        best_state = ft_ckpt.get("best_state")
        print(f"  Resumed from epoch {start_epoch}, best val UAUC={best_val_uauc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     epoch, kl_anneal_epochs=args.kl_anneal_epochs)

        val_preds = evaluate(model, val_loader, device)
        val_records = val_ds.records
        val_uauc = sequence_clustered_bootstrap(val_records, val_preds,
                                                 n_bootstrap=args.n_bootstrap,
                                                 seed=args.seed + epoch)

        print(f"  Epoch {epoch:2d}: train_loss={train_loss:.4f}, "
              f"val_UAUC={val_uauc['uauc_mean']:.4f} "
              f"[{val_uauc['uauc_ci95_low']:.4f}, {val_uauc['uauc_ci95_high']:.4f}]")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_uauc})

        if val_uauc["uauc_mean"] > best_val_uauc:
            best_val_uauc = val_uauc["uauc_mean"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                         if 'cls_conv' in k or 'cls_pred' in k}
            print(f"    → best val UAUC so far")

        # Checkpoint after every epoch (enables resume on timeout)
        torch.save({
            "epoch": epoch,
            "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()
                           if 'cls_conv' in k or 'cls_pred' in k},
            "optimizer_state": optimizer.state_dict(),
            "best_val_uauc": best_val_uauc,
            "best_state": best_state,
            "history": history,
        }, ckpt_path_ft)

    # ── Post-fine-tuning evaluation ─────────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)

    print(f"\nPost-fine-tuning evaluation (test set)...")
    preds_after = evaluate(model, test_loader, device)
    uauc_after = sequence_clustered_bootstrap(test_records, preds_after,
                                               n_bootstrap=args.n_bootstrap,
                                               seed=args.seed)
    print(f"  Post-FT UAUC: {uauc_after['uauc_mean']:.4f} "
          f"[{uauc_after['uauc_ci95_low']:.4f}, {uauc_after['uauc_ci95_high']:.4f}]")

    # ── Scale-stratified UAUC ───────────────────────────────────────────
    # Stratify by original bbox area to rule out scale confound
    scale_bins = {"small": (0, 200), "medium": (200, 1000), "large": (1000, 1e9)}
    scale_uauc = {}
    for bin_name, (lo, hi) in scale_bins.items():
        bin_records = [r for r in test_records
                      if lo <= (r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) *
                              (r["bbox_xyxy"][3] - r["bbox_xyxy"][1]) < hi]
        if len(bin_records) < 10:
            scale_uauc[bin_name] = None
            continue
        bin_ids = {r["crop_id"] for r in bin_records}
        bin_preds = [p for p, r in zip(preds_after, test_records) if r["crop_id"] in bin_ids]
        if len(bin_preds) < 10:
            scale_uauc[bin_name] = None
        else:
            scale_uauc[bin_name] = sequence_clustered_bootstrap(
                bin_records, bin_preds, n_bootstrap=args.n_bootstrap, seed=args.seed)
            print(f"  Scale {bin_name:>8s}: UAUC={scale_uauc[bin_name]['uauc_mean']:.4f} "
                  f"[{scale_uauc[bin_name]['uauc_ci95_low']:.4f}, "
                  f"{scale_uauc[bin_name]['uauc_ci95_high']:.4f}] "
                  f"(n={len(bin_records)})")

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "cond": cond_key,
        "label": label,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "kl_anneal_epochs": args.kl_anneal_epochs,
        },
        "dataset_splits": {
            "train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
            "train_tp": sum(1 for r in train_ds.records if r["is_tp"]),
            "train_fp": sum(1 for r in train_ds.records if not r["is_tp"]),
            "test_tp": sum(1 for r in test_ds.records if r["is_tp"]),
            "test_fp": sum(1 for r in test_ds.records if not r["is_tp"]),
        },
        "uauc_before_finetune": uauc_before,
        "uauc_after_finetune": uauc_after,
        "uauc_scale_stratified": scale_uauc,
        "history": history,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
    }

    # Pre-registered success criterion
    ci_low = uauc_after["uauc_ci95_low"]
    ci_high = uauc_after["uauc_ci95_high"]
    results["success_criterion_met"] = ci_low > 0.5
    results["success_summary"] = (
        f"Post-FT UAUC = {uauc_after['uauc_mean']:.4f} "
        f"[{ci_low:.4f}, {ci_high:.4f}]. "
        f"CI {'excludes' if ci_low > 0.5 else 'includes'} 0.5. "
        f"Criterion: {'MET' if ci_low > 0.5 else 'NOT MET'}."
    )

    print(f"\n{'='*70}")
    print(f"SUCCESS CRITERION: {results['success_summary']}")
    print(f"{'='*70}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"Results saved to: {results_path}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Fine-tune EDL head on balanced detection crops — Path B positive control"
    )
    parser.add_argument("--cond", type=str, default="d",
                       help="Condition: d (Average) or e (DS)")
    parser.add_argument("--crop-dir", type=str, default=CROP_DIR_DEFAULT,
                       help="Root directory with crop metadata from Step 1")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT,
                       help="Output directory for fine-tuning results")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--kl-anneal-epochs", type=int, default=10,
                       help="KL annealing epochs (≤ epochs)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splits")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Bootstrap resamples for CI")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.cond not in CONDITIONS:
        print(f"Unknown condition '{args.cond}'. Available: {list(CONDITIONS.keys())}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    finetune_and_evaluate(args.cond, CONDITIONS[args.cond], device, args)

    print("\nDone. Run analysis/e3_balanced_control.py --cond all to generate figures.")


if __name__ == "__main__":
    main()
