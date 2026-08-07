"""Forensic summary for the v19 Kaggle replay batch.

The older PowerShell analyzer focuses on generic skipped-action warnings.  This
script instead reconstructs the real top-level actions (an action in step N
answers the observation in step N-1), identifies our player, deduplicates by
EpisodeId, and prints the closing turns of every loss with card/attack names.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cg.api import all_attack, all_card_data


CARD_NAMES = {card.cardId: card.name for card in all_card_data()}
ATTACK_NAMES = {attack.attackId: attack.name for attack in all_attack()}
OPTION_NAMES = {
    0: "Number", 1: "Yes", 2: "No", 3: "Card", 4: "ToolCard",
    5: "EnergyCard", 6: "Energy", 7: "Play", 8: "Attach",
    9: "Evolve", 10: "Ability", 11: "Discard", 12: "Retreat",
    13: "Attack", 14: "End", 15: "Skill", 16: "SpecialCondition",
}

ARCHETYPES = [
    ("Dragapult", {119, 120, 121}),
    ("Alakazam", {741, 742, 743}),
    ("Grimmsnarl", {646, 647, 648}),
    ("Crustle", {344, 345}),
    ("Ogerpon/Hydrapple", {96, 149, 150}),
    ("Froslass", {860, 861}),
    ("Mega Lucario", {677, 678}),
    ("Gholdengo", {400, 401}),
    ("Typhlosion", {189, 190}),
]


def card_name(card_id: int | None) -> str:
    return CARD_NAMES.get(card_id, f"#{card_id}")


def load_replays(patterns: list[str]) -> list[tuple[Path, dict]]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    episodes = {}
    for path in sorted(set(paths)):
        if path.stat().st_size < 10_000:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episode = data.get("info", {}).get("EpisodeId")
        if episode is not None and data.get("steps"):
            episodes.setdefault(str(episode), (path, data))
    return list(episodes.values())


def our_index(data: dict) -> int:
    names = data.get("info", {}).get("TeamNames", [])
    exact = [index for index, name in enumerate(names) if name == "Rahul Sharma"]
    return exact[0] if exact else 0


def visible_ids(data: dict, player_index: int) -> set[int]:
    result = set()
    for step in data["steps"]:
        if player_index >= len(step):
            continue
        current = step[player_index].get("observation", {}).get("current")
        if not current:
            continue
        player = current["players"][1 - player_index]
        for area in ("active", "bench", "discard"):
            result.update(card["id"] for card in player.get(area, []) if card)
    return result


def archetype(ids: set[int]) -> str:
    matches = [name for name, signature in ARCHETYPES if ids & signature]
    return "+".join(matches) if matches else "Other"


def area_card(player: dict, area: int, index: int):
    key = {2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
    cards = player.get(key, []) if key else []
    return cards[index] if 0 <= index < len(cards) else None


def action_label(observation: dict, option: dict, acting_index: int) -> str:
    option_type = int(option.get("type", -1))
    label = OPTION_NAMES.get(option_type, f"Option{option_type}")
    current = observation["current"]
    player = current["players"][acting_index]
    card = None
    if option_type in {7, 8, 9}:
        card = area_card(player, 2, int(option.get("index", -1)))
    elif option_type in {10, 11}:
        card = area_card(player, int(option.get("area", -1)), int(option.get("index", -1)))
    if card:
        label += f" {card_name(card.get('id'))}"
    if option_type == 13:
        attack_id = option.get("attackId")
        label += f" {ATTACK_NAMES.get(attack_id, f'#{attack_id}')}"
    return label


def board(player: dict) -> str:
    active = next((card for card in player.get("active", []) if card), None)
    active_text = "none"
    if active:
        active_text = (
            f"{card_name(active['id'])} {active.get('hp', '?')}HP/"
            f"{len(active.get('energies', active.get('energyCards', [])))}E"
        )
    bench = ", ".join(
        f"{card_name(card['id'])} {card.get('hp', '?')}HP"
        for card in player.get("bench", []) if card
    ) or "empty"
    return f"active={active_text}; bench=[{bench}]"


def actions(data: dict) -> list[dict]:
    records = []
    for step_index in range(1, len(data["steps"])):
        previous = data["steps"][step_index - 1]
        current_step = data["steps"][step_index]
        for player_index in (0, 1):
            if player_index >= len(previous) or player_index >= len(current_step):
                continue
            observation = previous[player_index].get("observation", {})
            selected = current_step[player_index].get("action")
            select = observation.get("select")
            current = observation.get("current")
            if not current or not select or not selected:
                continue
            option_index = int(selected[0])
            options = select.get("option", [])
            if not 0 <= option_index < len(options):
                continue
            option = options[option_index]
            records.append({
                "step": step_index,
                "player": player_index,
                "turn": int(current.get("turn", -1)),
                "context": int(select.get("context", -1)),
                "label": action_label(observation, option, player_index),
                "players": current["players"],
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--closing-actions", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    replays = load_replays(args.patterns)
    matchup_results = defaultdict(Counter)
    losses = []
    wins = 0
    for path, data in replays:
        us = our_index(data)
        reward = int(data["rewards"][us])
        result = "WIN" if reward > 0 else "LOSS" if reward < 0 else "DRAW"
        matchup = archetype(visible_ids(data, us))
        matchup_results[matchup][result] += 1
        wins += result == "WIN"
        if result == "LOSS":
            losses.append((path, data, us, matchup, actions(data)))

    print(f"Unique valid episodes: {len(replays)}; wins={wins}; losses={len(losses)}")
    print("\nMatchups:")
    for name, results in sorted(matchup_results.items(), key=lambda item: -sum(item[1].values())):
        print(f"  {name:28} {results['WIN']:2}-{results['LOSS']:2}")

    print("\nLoss closing sequences:")
    for path, data, us, matchup, records in losses:
        episode = data["info"]["EpisodeId"]
        opponent = data["info"]["TeamNames"][1 - us]
        print(f"\n=== {episode} vs {opponent} [{matchup}] ===")
        selected_records = records[-args.closing_actions:]
        if args.compact:
            our_turns = [record["turn"] for record in records if record["player"] == us]
            op_turns = [record["turn"] for record in records if record["player"] != us]
            closing_turns = {
                (us, max(our_turns, default=-1)),
                (1 - us, max(op_turns, default=-1)),
            }
            selected_records = [
                record for record in records
                if (record["player"], record["turn"]) in closing_turns
                and (record["context"] == 0 or not record["label"].startswith("Card"))
            ]
        for record in selected_records:
            marker = "US" if record["player"] == us else "OP"
            me = record["players"][us]
            them = record["players"][1 - us]
            print(
                f"s{record['step']:03} t{record['turn']:02} {marker} "
                f"{record['label']}; prizes {len(me.get('prize', []))}-"
                f"{len(them.get('prize', []))}; our {board(me)}; opp {board(them)}"
            )


if __name__ == "__main__":
    main()
