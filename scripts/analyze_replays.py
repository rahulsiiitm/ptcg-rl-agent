"""
Analyze replays to extract opponent deck compositions.
Identifies which Pokemon IDs appear in opponent decks across all replays.
"""
import json
import os
from collections import Counter, defaultdict

REPLAY_DIR = "data/replays"
replays = [f for f in os.listdir(REPLAY_DIR) if f.endswith(".json") and "credentials" not in f]

import csv
id_to_name = {}
with open("data/EN_Card_Data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = int(row["Card ID"])
        id_to_name[cid] = row["Card Name"].strip()

opponent_decks = []
total_games = 0

for fname in replays:
    path = os.path.join(REPLAY_DIR, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Skip {fname}: {e}")
        continue

    # Replays can be a list of episodes or a single episode dict
    episodes = data if isinstance(data, list) else [data]
    
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        steps = ep.get("steps", []) or []
        
        # Find all card IDs used by each player across the game
        player_cards = defaultdict(set)
        
        for step in steps:
            if not isinstance(step, list):
                step = [step]
            for obs_wrapper in step:
                if not isinstance(obs_wrapper, dict):
                    continue
                obs = obs_wrapper.get("observation", obs_wrapper)
                current = obs.get("current", {}) if isinstance(obs, dict) else {}
                if not current:
                    continue
                players = current.get("players", [])
                for pi, player in enumerate(players):
                    if not isinstance(player, dict):
                        continue
                    # Collect from discard, active, bench
                    for area in ["discard", "active", "bench"]:
                        for card in (player.get(area) or []):
                            if isinstance(card, dict):
                                cid = card.get("id")
                                if cid:
                                    player_cards[pi].add(cid)
        
        if len(player_cards) >= 2:
            total_games += 1
            # player 0 is usually us (Lucario), player 1 is opponent
            # but let's collect both and filter out our own deck IDs
            our_ids = {673,674,675,676,677,678,344,345,1102,1123,1141,1142,1152,1159,1182,1192,1227,1252,6}
            for pi in [0, 1]:
                cards = player_cards[pi]
                if not cards.intersection(our_ids):  # not our deck
                    opponent_decks.append(cards)

print(f"Total episodes parsed: {total_games}")
print(f"Opponent deck samples: {len(opponent_decks)}")

# Find most common Pokemon in opponent decks
pokemon_counter = Counter()
for deck in opponent_decks:
    for cid in deck:
        if cid in id_to_name:
            pokemon_counter[cid] += 1

print("\n=== Most Common Opponent Cards ===")
for cid, count in pokemon_counter.most_common(40):
    print(f"  {count:3d}x  {cid:5d}: {id_to_name[cid]}")
