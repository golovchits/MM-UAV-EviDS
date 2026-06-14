#!/usr/bin/env python3
"""
Extract ALL frames from test-sequence tars for tracker evaluation.
Reads the tar_index.pkl, finds all frames belonging to the 121 test
sequences, and writes them as individual JPEG files.
"""
import pickle
import os

TAR_DIR = os.environ.get("TAR_DIR", "/path/to/MM-UAV-tars")
OUT_DIR = os.environ.get("OUT_DIR", "/path/to/MM-UAV-images")
INDEX_PATH = os.path.join(TAR_DIR, "tar_index.pkl")

# 121 test sequences (from the official MM-UAV benchmark split)
TEST_SEQS = [
    "0001","0003","0005","0023","0024","0025","0026","0042","0045","0046",
    "0047","0048","0049","0062","0063","0064","0065","0066","0067","0068",
    "0069","0070","0071","0104","0269","0270","0280","0281","0292","0319",
    "0320","0321","0322","0323","0324","0325","0326","0327","0328","0329",
    "0330","0331","0332","0335","0336","0339","0373","0406","0408","0409",
    "0427","0561","0562","0563","0564","0565","0566","0567","0568","0569",
    "0570","0571","0648","0714","0782","0788","0789","0793","0794","0801",
    "0802","0806","0909","0912","0920","0921","0922","0927","0928","0930",
    "0935","0937","0938","0939","0995","0996","0997","0998","0999","1014",
    "1017","1023","1060","1082","1083","1084","1085","1086","1087","1088",
    "1089","1090","1091","1092","1093","1097","1098","1099","1100","1101",
    "1200","1397","1811","1816","1836","1840","1842","1846","1859","1863",
    "1866",
]

print(f"Loading tar index from {INDEX_PATH} ...")
with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)
print(f"Index loaded: {len(index):,} entries")

# Filter to test sequences only
test_entries = {}
for relpath, (tar_path, offset, size) in index.items():
    # relpath format: "SEQ/rgb_frame/FRAME.jpg" or "SEQ/ir_frame/FRAME.jpg"
    parts = relpath.split("/", 1)
    seq = parts[0]
    if seq in TEST_SEQS:
        test_entries[relpath] = (tar_path, offset, size)

print(f"Test entries to extract: {len(test_entries):,}")

# Extract
extracted = 0
skipped = 0
open_handles = {}

for relpath, (tar_path, offset, size) in test_entries.items():
    out_path = os.path.join(OUT_DIR, relpath)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        skipped += 1
        continue

    # Read from tar via byte offset
    if tar_path in open_handles:
        fh = open_handles[tar_path]
    else:
        fh = open(tar_path, "rb")
        if len(open_handles) >= 16:
            oldest = next(iter(open_handles))
            open_handles[oldest].close()
            del open_handles[oldest]
        open_handles[tar_path] = fh

    fh.seek(offset)
    raw = fh.read(size)
    with open(out_path, "wb") as f:
        f.write(raw)
    extracted += 1

    if extracted % 5000 == 0:
        print(f"  {extracted:,}/{len(test_entries):,} extracted ...")

# Cleanup
for fh in open_handles.values():
    fh.close()

print(f"Done: {extracted:,} extracted, {skipped:,} already existed")

# Verify a few
for seq in TEST_SEQS[:3]:
    rgb_dir = os.path.join(OUT_DIR, seq, "rgb_frame")
    ir_dir = os.path.join(OUT_DIR, seq, "ir_frame")
    if os.path.isdir(rgb_dir):
        print(f"  {seq}/rgb_frame: {len(os.listdir(rgb_dir))} images")
    if os.path.isdir(ir_dir):
        print(f"  {seq}/ir_frame: {len(os.listdir(ir_dir))} images")
