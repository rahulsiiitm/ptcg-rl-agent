# AGENTS.md

## Project
RL/heuristic agent for the Kaggle "Pokémon TCG AI Battle Challenge - Simulation" 
competition (`pokemon-tcg-ai-battle`), running on the cabt engine 
(docs: https://matsuoinstitute.github.io/cabt/). Sibling Strategy-track competition 
(`…-challenge-strategy`) provides the card dataset and will later take a written report.

## Goal
Get a working agent on the real ladder ASAP, then improve it — in that order. 
Deadlines: entry Aug 9, 2026 / final submission Aug 16, 2026.

## Key evidence driving strategy (from public data on other teams' ladder results)
- Simple, consistent, single-prize decks piloted by a rule-based agent have 
  outperformed complex combo/ex decks in practice — a rule-based agent handles a 
  simple deck cleanly but plays a complex one clunkily.
- Local cabt-engine simulation has been shown to MISPREDICT real ladder rank. 
  The real ladder is the only reliable judge — local sim is a sanity check, not 
  ground truth.
- A prior RL+MCTS attempt on a strong deck regressed in ladder rating versus a 
  simpler heuristic scorer on the identical deck. More sophisticated ≠ better; 
  every change must be validated against real ladder results before being trusted.

## Build order (do not skip ahead)
1. **Phase 0 — Deck selection**: mine `data/EN_Card_Data.csv` for a simple, 
   consistent, single-prize deck. Document rationale in `decks/deck_rationale.md`.
2. **Phase 1 — Rule-based agent**: ship `src/agent/rule_based.py`, submit to the 
   REAL ladder, log the result. This is the fallback and the baseline everything 
   else must beat.
3. **Phase 2 — RL/opponent-modeling layer**: only begins after Phase 1 has a real 
   ladder score. Every RL agent version is A/B tested against the rule-based 
   baseline's actual ladder μ before being promoted to `main.py`.

## Constraints
- Submission = `main.py` (top-level, not nested) + `deck.csv`, tarballed via 
  `tar -czvf submission.tar.gz *`.
- Kaggle runtime: 2 vCPU, 12.2 GiB RAM, 11.8 GiB disk, submission size limit 197.7 MiB.
- Daily submission limit: 5; only the latest 2 stay active/scored on the ladder — 
  validate locally before every upload, don't waste slots.
- Engine only ever presents legal moves — action space size varies per turn 
  (variable-length legal move list). Policy/heuristics must score or rank the 
  given legal options, not assume a fixed discrete action space.
- Imperfect information: opponent's hand and remaining deck are hidden. State 
  encoding and heuristics must not assume knowledge of unseen cards, except 
  through `opponent_model.py`'s probabilistic estimate.
- 10-minute total time budget per match. Timing out = instant loss. Agent must 
  never crash — always return a legal fallback action.

## Dataset
`data/EN_Card_Data.csv` — official card metadata (~2000 cards, Standard format): 
Card ID, Card Name, Expansion, Collection No., Stage/Type, Rule, Category, 
Previous stage, HP, Type, Weakness, Resistance, Retreat, Move Name, Cost, Damage, 
Effect Explanation. Loaded via `data/card_lookup.py`. Do not redistribute this file 
outside the project — it's subject to competition rules.

## Architecture
- `src/env/wrapper.py` — Gymnasium-style wrapper around the cabt engine.
- `src/agent/rule_based.py` — Phase 1 heuristic agent (build/ship first).
- `src/agent/policy.py` — Phase 2 RL policy (PPO), built after Phase 1 ladders.
- `src/agent/opponent_model.py` — tracks played/discarded/revealed cards to 
  maintain a running probability estimate over the opponent's remaining deck/hand.
- `src/env/reward.py` — shaped reward (win/loss + prizes taken + KOs + board 
  advantage), isolated from the training loop for independent iteration.
- `src/train/self_play.py` — checkpoint pool + `rule_based.py` kept as a permanent 
  self-play opponent, to avoid overfitting to self-play only.

## Conventions
- Keep `main.py` a thin loader only — no training or heuristic logic there, just 
  imports + a saved/selected agent.
- Reward-shaping changes go only in `src/env/reward.py`.
- Run `scripts/validate_submission.sh` before every Kaggle upload.
- Log every real submission in `eval/ladder_log.md` (date, agent version, deck, 
  real ladder μ/Elo, notes).
- Log local-sim vs real-ladder result side by side in `eval/local_vs_ladder.md` 
  for every RL version, given the known mismatch risk.
- Never promote a Phase 2 agent to `main.py` unless its real ladder score beats 
  the Phase 1 rule-based baseline's real ladder score.

## Current status
Not started. Dataset not yet downloaded. Next action: download `EN_Card_Data.csv` 
from the competition's dataset tab into `data/`, then begin Phase 0.