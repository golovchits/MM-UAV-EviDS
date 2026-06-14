#!/usr/bin/env python3
"""Extract only annotation-referenced images from tar shards to individual files.

Reads train-rgb.json, train-ir.json, val-rgb.json, val-ir.json from
annotations/ dir, finds each referenced image in the tar index, extracts
it to an output directory mirrored structure.

Usage: python extract_images.py
  Reads from: $TAR_DIR (default: /path/to/MM-UAV-tars/)
  Writes to:  $OUT_DIR (default: /path/to/MM-UAV-images/)
"""

import json
import os
import sys
from pathlib import Path

TAR_DIR = Path(os.environ.get("TAR_DIR", "/path/to/MM-UAV-tars"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/path/to/MM-UAV-images"))
ANNOTATIONS = TAR_DIR / "annotations"

JSON_FILES = ["train-rgb.json", "train-ir.json", "val-rgb.json", "val-ir.json"]


def load_tar_index():
    """Load the merged tar byte-offset index from pickle cache or JSON files."""
    import pickle

    cache_path = TAR_DIR / "tar_index.pkl"
    if cache_path.exists():
        print(f"Loading tar index from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Pickle cache not found, building from JSON indexes...")
    index = {}
    for idx_file in sorted(TAR_DIR.glob("*.idx.json")):
        tar_name = idx_file.stem.replace(".idx", "")
        tar_path = TAR_DIR / f"{tar_name}.tar"
        if not tar_path.exists():
            continue
        with open(idx_file) as f:
            seq_index = json.load(f)
        for relpath, (offset, size) in seq_index.items():
            index[relpath] = (str(tar_path), offset, size)
    print(f"Built index: {len(index):,} entries")
    return index


def main():
    import cv2
    import numpy as np

    print("Loading tar index...")
    index = load_tar_index()
    print(f"Index loaded: {len(index):,} entries")

    # Collect all unique file paths from annotations
    all_files = set()
    for json_name in JSON_FILES:
        json_path = ANNOTATIONS / json_name
        if not json_path.exists():
            print(f"WARNING: {json_path} not found, skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        for img in data["images"]:
            all_files.add(img["file_name"])
        print(f"  {json_name}: {len(data['images'])} images")

    print(f"\nTotal unique images to extract: {len(all_files)}")

    # Try prefix stripping for tar index lookup
    # Annotations have "train/0001/..." but tar index has "0001/..."
    not_found = 0
    extracted = 0
    skipped = 0

    # Collect all index keys for prefix matching
    index_lookup = {}
    for key in index:
        # Store without prefix for matching
        parts = key.split("/", 1)
        if len(parts) > 1:
            clean = parts[1]
        else:
            clean = key
        index_lookup[clean] = key

    for file_name in sorted(all_files):
        # Try exact match first
        entry = index.get(file_name)
        if entry is None:
            # Try without first path component
            parts = file_name.split("/", 1)
            clean = parts[1] if len(parts) > 1 else file_name
            entry = index.get(clean)

        if entry is None:
            not_found += 1
            if not_found <= 5:
                print(f"  NOT FOUND: {file_name}")
            continue

        tar_path, offset, size = entry

        # Determine output path: strip the dataset prefix (e.g., "train/")
        out_rel = file_name.split("/", 1)[1] if "/" in file_name else file_name
        out_path = OUT_DIR / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            skipped += 1
            continue

        # Read from tar
        try:
            with open(tar_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  DECODE FAILED: {file_name}")
                continue
            cv2.imwrite(str(out_path), img)
            extracted += 1
        except Exception as e:
            print(f"  ERROR: {file_name}: {e}")
            continue

        if extracted % 5000 == 0:
            print(f"  Progress: {extracted} extracted, {skipped} skipped...")

    print(f"\nDone: {extracted} extracted, {skipped} skipped (already exist), {not_found} not found")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
