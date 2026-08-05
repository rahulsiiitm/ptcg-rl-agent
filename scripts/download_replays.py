#!/usr/bin/env python3
"""
scripts/download_replays.py

Downloads the most recent PTCG AI Battle episode datasets from Kaggle and
extracts them to data/replays/. Run once before training to prime the
replay buffer with real high-level gameplay data.

Usage:
    python scripts/download_replays.py [--days 3]

Requires:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - kaggle package installed (pip install kaggle)
"""
import argparse
import os
import subprocess
import sys
import zipfile
from datetime import date, timedelta

REPLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")

def get_recent_dataset_refs(num_days: int = 4) -> list[str]:
    """Generates dataset refs for the last N days."""
    refs = []
    today = date.today()
    for i in range(num_days):
        d = today - timedelta(days=i)
        refs.append(f"kaggle/pokemon-tcg-ai-battle-episodes-{d.strftime('%Y-%m-%d')}")
    return refs

def download_dataset(ref: str, out_dir: str) -> bool:
    """Downloads and unzips a dataset. Returns True on success."""
    slug = ref.split("/")[-1]
    dest = os.path.join(out_dir, slug)
    if os.path.exists(dest) and len(os.listdir(dest)) > 0:
        print(f"[SKIP] {slug} already downloaded.")
        return True

    os.makedirs(dest, exist_ok=True)
    print(f"[DOWNLOAD] {ref} ...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", ref, "-p", dest, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  -> FAILED: {result.stderr.strip()}")
        # Cleanup empty dir
        try:
            os.rmdir(dest)
        except OSError:
            pass
        return False

    print(f"  -> OK ({len(os.listdir(dest))} files)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download recent PTCG Kaggle replay datasets.")
    parser.add_argument("--days", type=int, default=4, help="How many recent days to download (default: 4).")
    args = parser.parse_args()

    os.makedirs(REPLAY_DIR, exist_ok=True)
    refs = get_recent_dataset_refs(args.days)

    success, fail = 0, 0
    for ref in refs:
        if download_dataset(ref, REPLAY_DIR):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} downloaded, {fail} failed.")
    print(f"Replays stored in: {REPLAY_DIR}")
    print("\nNext step: run scripts/parse_replays_to_training.py to convert replays to training data.")

if __name__ == "__main__":
    main()
