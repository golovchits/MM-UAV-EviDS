#!/usr/bin/env python3
"""Create tar shards with byte-offset indexes for O(1) random access.

Usage:
    python create_tars.py \
        --data_dir /Volumes/My\ Passport/MM-UAV-extracted/MMMUAV/ \
        --output_dir /Volumes/My\ Passport/MM-UAV-tars/ \
        --split train \
        --max_sequences 0  # 0 = all

Two-pass approach:
  1. Create tar per sequence with tarfile (no compression)
  2. Read tar back in stream mode to build byte-offset index (.idx.json)

Output structure:
  MM-UAV-tars/
    annotations/
      train-rgb.json, train-ir.json, val-rgb.json, val-ir.json
    <seq_id>.tar
    <seq_id>.idx.json
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def get_sequences(data_dir, split):
    """List sequence directories in the given split."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return []
    return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])


def create_tar_for_sequence(seq_id, data_dir, output_dir, split):
    """Create tar file for a single sequence."""
    seq_path = Path(data_dir) / split / seq_id
    tar_path = Path(output_dir) / f"{seq_id}.tar"
    idx_path = Path(output_dir) / f"{seq_id}.idx.json"

    # Resume: skip if both tar and valid index already exist
    if tar_path.exists() and idx_path.exists():
        try:
            with open(idx_path) as f:
                idx = json.load(f)
            if idx:
                return seq_id, 0, "skipped"
        except (json.JSONDecodeError, OSError):
            pass
        # Corrupt index — remove both and redo
        tar_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)
    elif tar_path.exists() and not idx_path.exists():
        # Partial tar from interrupted run — remove and redo
        tar_path.unlink(missing_ok=True)

    # Collect files to archive
    files_to_add = []
    for subdir in ["rgb_frame", "ir_frame", "event_frame"]:
        subdir_path = seq_path / subdir
        if subdir_path.exists():
            for f in sorted(subdir_path.iterdir()):
                if f.is_file():
                    # Store as relative path: <seq_id>/rgb_frame/frame_000001.png
                    arcname = f"{seq_id}/{subdir}/{f.name}"
                    files_to_add.append((str(f), arcname))

    # Ground-truth and metadata files
    for pattern in ["gt_rgb/gt.txt", "gt_ir/gt.txt", "seqinfo-rgb.ini",
                    "seqinfo-ir.ini", "seqinfo-event.ini"]:
        p = seq_path / pattern
        if p.exists():
            arcname = f"{seq_id}/{pattern}"
            files_to_add.append((str(p), arcname))

    if not files_to_add:
        return seq_id, 0, None

    # Pass 1: create tar
    with tarfile.open(tar_path, 'w') as tar:
        for file_path, arcname in files_to_add:
            tar.add(file_path, arcname=arcname)

    # Pass 2: build index by reading tar back in stream mode
    index = {}
    with tarfile.open(tar_path, 'r|') as tar:
        for member in tar:
            # member.offset_data is available in Python 3.12+
            # For older Python, use member.offset + 512 (standard header size)
            data_offset = getattr(member, 'offset_data', member.offset + 512)
            index[member.name] = [data_offset, member.size]

    # Save index
    idx_path = Path(output_dir) / f"{seq_id}.idx.json"
    with open(idx_path, 'w') as f:
        json.dump(index, f)

    tar_size = tar_path.stat().st_size / (1024 ** 3)  # GB
    return seq_id, len(files_to_add), tar_size


def process_sequence(args):
    """Wrapper for multiprocessing."""
    seq_id, data_dir, output_dir, split = args
    try:
        return create_tar_for_sequence(seq_id, data_dir, output_dir, split)
    except Exception as e:
        return seq_id, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Create tar shards with byte-offset indexes")
    parser.add_argument("--data_dir", required=True,
                        help="Path to extracted MM-UAV data (MMMUAV/)")
    parser.add_argument("--output_dir", required=True,
                        help="Path to output tar directory")
    parser.add_argument("--split", default="train",
                        choices=["train", "test"],
                        help="Dataset split to process")
    parser.add_argument("--max_sequences", type=int, default=0,
                        help="Max sequences to process (0 = all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy annotations
    ann_src = Path(args.data_dir) / "annotations"
    ann_dst = Path(args.output_dir) / "annotations"
    if ann_src.exists() and not ann_dst.exists():
        print(f"Copying annotations from {ann_src} to {ann_dst}...")
        shutil.copytree(ann_src, ann_dst)

    # Get sequences
    sequences = get_sequences(args.data_dir, args.split)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]

    print(f"Found {len(sequences)} sequences in {args.split} split")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")

    total_files = 0
    total_size_gb = 0.0
    errors = []
    t_start = time.time()

    if args.workers > 1:
        tasks = [(s, args.data_dir, args.output_dir, args.split) for s in sequences]
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_sequence, tasks),
                total=len(tasks),
                desc="Creating tars"
            ))
    else:
        results = []
        for seq_id in tqdm(sequences, desc="Creating tars"):
            results.append(create_tar_for_sequence(seq_id, args.data_dir,
                                                    args.output_dir, args.split))

    skipped = 0
    for seq_id, n_files, size_or_err in results:
        if isinstance(size_or_err, str):
            if size_or_err == "skipped":
                skipped += 1
            else:
                errors.append((seq_id, size_or_err))
        else:
            total_files += n_files
            total_size_gb += size_or_err

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 3600:.1f} hours ({elapsed / 60:.1f} minutes)")
    print(f"Total files archived: {total_files:,}")
    print(f"Total tar size: {total_size_gb:.1f} GB")
    print(f"Skipped (already done): {skipped}")
    print(f"Newly created: {len(results) - len(errors) - skipped}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for seq_id, err in errors[:10]:
            print(f"  {seq_id}: {err}")


if __name__ == "__main__":
    main()
