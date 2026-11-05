"""Download a small, feasible FRED subset (smallest train/test sequences).

FRED is ~204 GB in total and its raw event .hdf5 requires the proprietary
Prophesee/Metavision SDK. We therefore use the *rendered event-frame PNGs*
included in each sequence zip, and treat each recording sequence as one
"place" (scene) for a sequence-retrieval objective.

Only the smallest sequences are fetched so the download stays tractable.
"""

from __future__ import annotations

import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

REPO = "GabrieleMagrini/FRED"
ROOT = os.path.join(os.path.dirname(__file__), "data", "fred")

# Canonical split: train has 184 sequences, test has 47. Use a small,
# disk-safe subset of the smallest sequences (each sequence is one "scene").
TRAIN_IDS = ["36", "108", "67", "35"]
TEST_IDS = ["21", "111", "115", "65"]


def download_one(split: str, seq_id: str) -> str:
    os.makedirs(os.path.join(ROOT, split), exist_ok=True)
    dest = os.path.join(ROOT, split, f"{seq_id}.zip")
    if os.path.exists(dest):
        print(f"[skip] {split}/{seq_id}.zip exists")
        return dest
    print(f"[download] {split}/{seq_id}.zip ...")
    hf_hub_download(
        repo_id=REPO,
        filename=f"{split}/{seq_id}.zip",
        repo_type="dataset",
        local_dir=ROOT,
    )
    return dest


def extract_one(split: str, seq_id: str, zip_path: str) -> str:
    # sequence zips contain a top-level dir named after the sequence id
    out_dir = os.path.join(ROOT, "extracted", split, seq_id)
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, seq_id, "Event", "Frames")
    if os.path.isdir(marker):
        print(f"[skip] extracted {split}/{seq_id}")
        return marker
    print(f"[extract] {split}/{seq_id} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return marker


def main() -> None:
    for split, ids in (("train", TRAIN_IDS), ("test", TEST_IDS)):
        for seq_id in ids:
            zp = download_one(split, seq_id)
            extract_one(split, seq_id, zp)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())