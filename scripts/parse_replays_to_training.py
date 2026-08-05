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

def load_replay_files(limit: int) -> list[dict]:
    """Loads replay JSON files from REPLAY_DIR."""
    all_files = []
    for root, dirs, files in os.walk(REPLAY_DIR):
        for fn in files:
            if fn.endswith(".json"):
                all_files.append(os.path.join(root, fn))

    if not all_files:
        print(f"No replay files found in {REPLAY_DIR}. Run download_replays.py first.")
        sys.exit(1)

    print(f"Found {len(all_files)} replay files.")
    episodes = []
    for fp in all_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            batch = data if isinstance(data, list) else [data]
            episodes.extend(batch)
        except Exception as e:
            print(f"  [SKIP] {fp}: {e}")
        if limit and len(episodes) >= limit:
            break

    return episodes[:limit] if limit else episodes

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

                sv_enc = get_encoder_input(obs)
                sv_dec = get_decoder_input(obs)

                # Build a one-hot policy from the actual action taken
                policy = [0.0] * n_options
                for a in (action if isinstance(action, list) else [action]):
                    if isinstance(a, int) and 0 <= a < n_options:
                        policy[a] = 1.0
                if sum(policy) == 0:
                    continue
                # Normalize
                total = sum(policy)
                policy = [p / total for p in policy]

                # Label: winner's moves get +1, loser's get -1
                player_idx = obs.current.yourIndex
                value = 1.0 if player_idx == winner_idx else -1.0

                s = LearnSample()
                s.sv_enc = sv_enc
                s.sv_dec = sv_dec
                s.policy = policy
                s.value = value
                samples.append(s)

            except Exception:
                continue

    return samples

def main():
    parser = argparse.ArgumentParser(description="Parse Kaggle PTCG replays into pretraining data.")
    parser.add_argument("--limit", type=int, default=5000, help="Max number of episodes to parse (default 5000).")
    parser.add_argument("--out", type=str, default="data/pretrain.pt", help="Output path for training data.")
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    episodes = load_replay_files(args.limit)
    print(f"Parsing {len(episodes)} episodes ...")

    all_samples = []
    for i, ep in enumerate(episodes):
        samples = parse_episode_to_samples(ep)
        all_samples.extend(samples)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(episodes)} episodes -> {len(all_samples)} samples so far")

    print(f"\nTotal training samples: {len(all_samples)}")
    torch.save(all_samples, out_path)
    print(f"Saved to {out_path}")
    print("\nNext step: update train_hybrid_mcts.py to load data/pretrain.pt at startup.")

if __name__ == "__main__":
    main()
