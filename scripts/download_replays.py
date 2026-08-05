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
import shutil
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
    """Downloads a dataset as a zip file. Returns True on success."""
    slug = ref.split("/")[-1]
    dest = os.path.join(out_dir, slug)

    # Check if we already have the downloaded zip
    expected_zip = os.path.join(out_dir, f"{slug}.zip")
    if os.path.exists(expected_zip):
        print(f"[SKIP] {slug} already downloaded.")
        return True

    os.makedirs(dest, exist_ok=True)
    
    print(f"[DOWNLOAD] {ref} ...")
    try:
        # Download as zip without --unzip
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", ref, "-p", dest],
            capture_output=True, text=True, timeout=600
        )
    except KeyboardInterrupt:
        print("\n[CANCELLED] Download interrupted by user.")
        raise
    except subprocess.TimeoutExpired:
        print(f"  -> TIMEOUT after 10 minutes.")
        shutil.rmtree(dest, ignore_errors=True)
        return False

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "403" in stderr or "Forbidden" in stderr:
            print(f"  -> SKIP (403 Forbidden — dataset not yet released or requires competition acceptance)")
        else:
            print(f"  -> FAILED: {stderr}")
        shutil.rmtree(dest, ignore_errors=True)
        return False

    # Find the downloaded zip file and move it to the parent directory
    zips = [f for f in os.listdir(dest) if f.endswith('.zip')]
    if not zips:
        print(f"  -> No zip found after download.")
        shutil.rmtree(dest, ignore_errors=True)
        return False

    downloaded_zip = os.path.join(dest, zips[0])
    target_zip = os.path.join(out_dir, f"{slug}.zip")
    shutil.move(downloaded_zip, target_zip)
    
    # Remove the temporary download folder
    shutil.rmtree(dest, ignore_errors=True)
    
    print(f"  -> OK ({os.path.getsize(target_zip)//1024**2} MB zip saved)")
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
