# Deck Selection Rationale

Based on Phase 0 guidelines, we need simple, consistent, single-prize decks that an AI can pilot effectively without complex combo requirements.

## Phase 1 & 2: Mono-Water Aggro
Originally, we used a simple Mono-Water basic deck to ensure the rule-based agent and early RL iterations had an easy time piloting it (just attach {W} and attack). This provided our baseline score of 170.1 (Phase 1) and 303.6 (Phase 2).

## Phase 3: Snorlax Control (FAILED — Bench Out Bug)
For Phase 3, we upgraded to a **Snorlax Control** deck (4 Pokémon only). The deck itself was never illegal — but only having 4 Pokémon meant the agent lost instantly every time its Active was KO'd with an empty bench. This "Bench Out" rule caused a **massive regression to 154.6 Elo**.

**Lesson**: Minimum viable bench depth is 8–12 Pokémon. Never run fewer than 8.

---

## Phase 4: Iono's Bellibolt ex / Kilowattrel (ACTIVE)

### Why this deck?
Based on public Kaggle ladder data, the **Iono's Bellibolt ex** archetype achieved the highest measured ladder Elo of **836** — the only deck with a clean answer to Crustle (~50% of the meta), which completely walls all ex Pokémon damage.

### Deck List (`deck.csv`):
```
# Pokemon (12)
4x Iono's Tadbulb (268)        — 60 HP Basic, benches easily via Poffin
4x Iono's Bellibolt ex (269)   — 280 HP Stage 1; Electric Streamer (Ability: free {L} attach from hand)
                                  Thunderous Bolt: 230 damage, 1-turn cooldown
2x Iono's Wattrel (270)        — 60 HP Basic, partner for Kilowattrel
2x Iono's Kilowattrel (271)    — 120 HP Stage 1; Flashing Draw (discard {L} → draw to 6)
                                  Mach Bolt: 70 damage — NON-EX, bypasses Crustle immunity

# Trainers (28)
4x Buddy-Buddy Poffin (1086)   — bench 2 Basics ≤70 HP from deck (fills bench T1)
4x Ultra Ball (1121)           — discard 2, search any Pokémon
4x Boss's Orders (1182)        — Supporter: pull opponent bench target active
4x Cheren (1224)               — Supporter: draw 3 cards
4x Switch (1123)               — retreat tool for Bellibolt cooldown turns
4x Pokémon Catcher (1124)      — Item gust (coinflip)
1x Master Ball (1125)          — ACE SPEC (MAX 1 COPY): search any Pokémon
1x Love Ball (1083)            — search Pokémon matching one you have in play

# Energy (20)
20x Basic {L} Lightning Energy (4)
```

### ACE SPEC Rule (Important!)
`Master Ball (1125)` is an **ACE SPEC** card. Decks may run a **maximum of 1 ACE SPEC card total**. Running 2 caused a "deck error" rejection on Kaggle. The second slot was replaced with `Love Ball (1083)`.

### Agent Design: `src/agent/rule_based_bellibolt.py`
Rather than relying on RL to discover the optimal strategy, we implemented a hardcoded **priority-ordered rule-based agent** matching the PDF spec:

| Priority | Action | Reason |
|---|---|---|
| 1 | Electric Streamer (Ability) | Attach ALL {L} from hand first — maximizes energy acceleration |
| 2 | Evolve | Always evolve when legal (engine blocks T1 evolve automatically) |
| 3 | Attach energy (manual) | Secondary to Streamer |
| 4 | Supporter: Boss > Cheren | Boss only if bench target exists; Cheren if hand ≤ 5 |
| 5 | Items: UBall > Poffin > MBall > LoveBall > Catcher | Search for missing pieces first |
| 6 | Retreat if needed | Bellibolt on cooldown OR Crustle matchup → bring in Kilowattrel |
| 7 | Attack | Thunderous Bolt (if not cooldown) or Mach Bolt (Kilowattrel) |

### Crustle Matchup (§3.3 Key Logic)
`Crustle` blocks all damage from Pokémon ex. Bellibolt ex attacks are suppressed.
The agent detects Crustle by card ID (`{348, 349}`) or name substring `"crustle"`.
When detected: blocks the attack branch entirely → retreats Bellibolt → sends in Kilowattrel (non-ex, Mach Bolt connects).

### Bellibolt Cooldown (§4 Key Logic)
`Thunderous Bolt` has a 1-turn "can't attack" clause. The cabt engine enforces this by simply **not presenting Thunderous Bolt as a legal option** the following turn. The agent detects cooldown by checking if "thunderous" is absent from attack options — no manual state tracking needed.

### Why Rule-Based > RL Here
From `AGENTS.md`: *"More sophisticated ≠ better. A simpler heuristic scorer on the identical deck outperformed RL."* The Bellibolt strategy is a small, deterministic decision tree:
- 2 attacker choices (Bellibolt / Kilowattrel)
- 1 conditional branch (Crustle present?)
- 1 cooldown check

PPO would need thousands of episodes to rediscover this by reward signal. The rule-based agent executes it perfectly from turn 1.

### Submission Bug History
| Attempt | Error | Fix |
|---|---|---|
| Phase 4 v1 | Import error (missing `__init__.py`) | Added `src/__init__.py`, `src/agent/__init__.py`, `data/__init__.py` |
| Phase 4 v2 | "Player 1's deck error" | `Master Ball` is ACE SPEC — max 1 copy. Replaced 2nd with `Love Ball`. |
| Phase 4 v3 | ✅ Submitted successfully | — |
