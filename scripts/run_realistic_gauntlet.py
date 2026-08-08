"""Run repeatable cabt matches against a standalone historical submission.

This is a ladder sanity check, not a ladder predictor.  Both players use the
real cabt engine and their actual submission policies; seats alternate to
remove the first-player bias.  A compact tail of every lost game is written as
JSON for diagnosis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from src.agent import rule_based_lucario as candidate_module  # noqa: E402


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("historical_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load historical agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path} contains {len(cards)} cards, expected 60")
    return cards


def _reset(agent) -> None:
    agent({"select": None})


def _card_summary(card):
    if card is None:
        return None
    return {
        "id": card.get("id"),
        "hp": card.get("hp"),
        "energy": len(card.get("energyCards") or card.get("energies") or []),
    }


def _state_summary(obs: dict) -> dict:
    current = obs.get("current") or {}
    players = []
    for player in current.get("players") or []:
        players.append({
            "deck": player.get("deckCount"),
            "hand": player.get("handCount", len(player.get("hand") or [])),
            "prizes": len(player.get("prize") or []),
            "active": [_card_summary(card) for card in player.get("active") or []],
            "bench": [_card_summary(card) for card in player.get("bench") or []],
        })
    return {
        "turn": current.get("turn"),
        "acting_player": current.get("yourIndex"),
        "result": current.get("result"),
        "players": players,
    }


def _option_summary(option: dict) -> dict:
    return {
        key: option.get(key)
        for key in ("type", "attackId", "area", "index", "inPlayArea", "inPlayIndex")
        if key in option
    }


def play_game(decks, agents, candidate_seat: int, max_decisions: int) -> dict:
    for agent in agents:
        _reset(agent)
    obs, start = battle_start(*decks)
    if start.errorType != 0 or obs is None:
        raise RuntimeError(f"cabt start failed: player={start.errorPlayer} type={start.errorType}")
    trace = []
    try:
        for decision in range(max_decisions):
            current = obs.get("current")
            if current is None or current.get("result", -1) != -1:
                break
            seat = current["yourIndex"]
            action = agents[seat](obs)
            options = (obs.get("select") or {}).get("option") or []
            trace.append({
                "decision": decision,
                "seat": seat,
                "context": (obs.get("select") or {}).get("context"),
                "action": action,
                "chosen": [_option_summary(options[index]) for index in action if 0 <= index < len(options)],
                "state": _state_summary(obs),
            })
            obs = battle_select(action)
        else:
            raise RuntimeError(f"game exceeded {max_decisions} decisions")
        result = (obs.get("current") or {}).get("result")
        return {
            "candidate_seat": candidate_seat,
            "winner": result,
            "candidate_won": result == candidate_seat,
            "decisions": len(trace),
            "terminal": _state_summary(obs),
            "tail": trace[-20:] if result != candidate_seat else [],
        }
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--opponent", type=Path, default=ROOT / "v15" / "main.py")
    parser.add_argument("--opponent-deck", type=Path, default=ROOT / "v15" / "deck.csv")
    parser.add_argument("--candidate-deck", type=Path, default=ROOT / "deck.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "v26_local_gauntlet.json")
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--disable-candidate-search", action="store_true")
    args = parser.parse_args()

    if args.disable_candidate_search:
        candidate_module.USE_SEARCH = False
    candidate_agent = candidate_module.agent
    opponent_agent = _load_agent(args.opponent.resolve())
    candidate_deck = _deck(args.candidate_deck.resolve())
    candidate_module.my_deck = list(candidate_deck)
    opponent_deck = _deck(args.opponent_deck.resolve())
    games = []
    for game_index in range(args.games):
        candidate_seat = game_index % 2
        if candidate_seat == 0:
            decks = (candidate_deck, opponent_deck)
            agents = (candidate_agent, opponent_agent)
        else:
            decks = (opponent_deck, candidate_deck)
            agents = (opponent_agent, candidate_agent)
        game = play_game(decks, agents, candidate_seat, args.max_decisions)
        game["game"] = game_index + 1
        games.append(game)
        print(
            f"game={game_index + 1} seat={candidate_seat} "
            f"winner={game['winner']} decisions={game['decisions']}"
        )

    wins = sum(game["candidate_won"] for game in games)
    report = {
        "candidate": "v26",
        "opponent": str(args.opponent),
        "games": games,
        "summary": {"wins": wins, "losses": len(games) - wins, "games": len(games)},
        "warning": "Local cabt is a sanity check; the real Kaggle ladder is ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
