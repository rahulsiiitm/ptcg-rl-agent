#!/usr/bin/env python3
"""
scripts/parse_replays_to_training.py

Parses downloaded Kaggle PTCG replay JSONs and converts them into
LearnSample-compatible training data that can be directly loaded into
the Experience Replay Buffer in train_hybrid_mcts.py.

This is "Imitation Learning" (Behavioral Cloning) -- we learn from
real high-quality games rather than starting from scratch via self-play.

Usage:
    python scripts/parse_replays_to_training.py [--limit 1000] [--out data/pretrain.pt]

Output:
    data/pretrain.pt  --  a list[LearnSample] saved with torch.save()
"""
import argparse
import json
import os
import sys
import glob

import torch

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
_cg_paths = glob.glob('/kaggle/input/**/cg-lib', recursive=True)
if _cg_paths:
    sys.path.append(_cg_paths[0])
elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cg')):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cg.api import to_observation_class
from src.agent.hybrid_lucario import LearnSample, get_encoder_input, get_decoder_input, SparseVector

REPLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")

def load_replay_files(limit: int = None):
    """Loads replay JSON files from REPLAY_DIR (including those inside .zip files)."""
    import zipfile
    count = 0
    for root, dirs, files in os.walk(REPLAY_DIR):
        for fn in sorted(files):
            if fn.endswith(".json"):
                # Handle single unzipped JSON (rare in this dataset)
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"  [SKIP] {path}: {e}")
                        continue
                batch = data if isinstance(data, list) else [data]
                for ep in batch:
                    yield ep
                    count += 1
                    if limit and count >= limit:
                        return
                        
            if fn.endswith(".zip"):
                zip_path = os.path.join(root, fn)
                print(f"Scanning zip: {fn}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        for zinfo in z.infolist():
                            if zinfo.filename.endswith(".json"):
                                try:
                                    with z.open(zinfo) as f:
                                        data = json.load(f)
                                    batch = data if isinstance(data, list) else [data]
                                    for ep in batch:
                                        yield ep
                                        count += 1
                                        if count > 0 and count % 10 == 0:
                                            print(f"  -> Extracted {count} valid replays so far...")
                                        if limit and count >= limit:
                                            return
                                except Exception as e:
                                    pass # Corrupt JSON inside zip
                except zipfile.BadZipFile:
                    print(f"  [SKIP] {fn}: Bad Zip File")

    if count == 0:
        print(f"No replay data found in {REPLAY_DIR}. Run download_replays.py first.")
        sys.exit(1)

deck_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deck.csv")
with open(deck_path, "r", encoding="utf-8") as f:
    LUCARIO_DECK = [int(line) for line in f.read().splitlines() if line.strip()]

def parse_episode_to_samples(ep: dict) -> list[LearnSample]:
    """
    Converts a single replay episode to a list of LearnSamples.
    The winner's moves are labeled 1.0 (win) and the loser's -1.0 (loss).
    This is behavioral cloning — we teach the model to copy winning moves.
    """
    samples = []
    steps = ep.get("steps", [])
    rewards = ep.get("rewards", [None, None])

    # Determine winner: rewards[i] > 0 means player i won
    winner_idx = None
    for i, r in enumerate(rewards):
        if r is not None and r > 0:
            winner_idx = i
            break

    if winner_idx is None:
        return []  # No clear winner (draw / null), skip

    for step in steps:
        if not isinstance(step, list):
            step = [step]
        for obs_wrapper in step:
            if not isinstance(obs_wrapper, dict):
                continue
            observation = obs_wrapper.get("observation")
            action = obs_wrapper.get("action")
            if observation is None or action is None:
                continue

            try:
                obs = to_observation_class(observation)
                if obs.select is None:
                    continue
                n_options = len(obs.select.option)
                if n_options == 0:
                    continue

                actions_list = []
                indices = list(range(obs.select.maxCount))
                for _ in range(64):
                    actions_list.append(indices.copy())
                    for i in range(len(indices)):
                        index = len(indices) - i - 1
                        if indices[index] < len(obs.select.option) - i - 1:
                            indices[index] += 1
                            for j in range(index+1, len(indices)):
                                indices[j] = indices[j - 1] + 1
                            break
                    else:
                        break

                sv_enc = get_encoder_input(obs, LUCARIO_DECK)
                sv_dec = get_decoder_input(obs, actions_list)

                # Build a one-hot policy from the actual action taken
                policy = [0.0] * len(actions_list)
                
                # The Kaggle 'action' is what they actually chose. We need to match it to actions_list.
                # Kaggle action could be an int or a list of ints.
                actual_action = action if isinstance(action, list) else [action]
                
                match_idx = -1
                for idx, act in enumerate(actions_list):
                    if act == actual_action:
                        match_idx = idx
                        break
                
                if match_idx == -1:
                    continue # Action not found in the up-to-64 generated list
                
                policy[match_idx] = 1.0
                # Normalize
                total = sum(policy)
                policy = [p / total for p in policy]

                # Label: winner's moves get +1, loser's get -1
                player_idx = obs.current.yourIndex
                value = 1.0 if player_idx == winner_idx else -1.0

                s = LearnSample(value=value, policy=policy, sv_enc=sv_enc, sv_dec=sv_dec)
                samples.append(s)

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

    return samples

def main():
    parser = argparse.ArgumentParser(description="Parse Kaggle PTCG replays into pretraining data.")
    parser.add_argument("--limit", type=int, default=5000, help="Max number of episodes to parse (default 5000).")
    parser.add_argument("--out", type=str, default="data/pretrain.pt", help="Output path for training data.")
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    episodes_generator = load_replay_files(args.limit)
    print(f"Parsing episodes up to limit {args.limit} ...")

    all_samples = []
    for i, ep in enumerate(episodes_generator):
        samples = parse_episode_to_samples(ep)
        all_samples.extend(samples)
        if (i + 1) % 100 == 0:
            print(f"  {i+1} episodes -> {len(all_samples)} samples so far")

    print(f"\nTotal training samples: {len(all_samples)}")
    torch.save(all_samples, out_path)
    print(f"Saved to {out_path}")
    print("\nNext step: update train_hybrid_mcts.py to load data/pretrain.pt at startup.")

if __name__ == "__main__":
    main()
