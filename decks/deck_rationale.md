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

### Phase 5 - 10: The Deep RL Pivot (Snorlax/Lopunny Control)
While the rule-based Bellibolt agent was stable, it was fundamentally limited by its inability to adapt to complex board states. To push past the 303.6 peak, we pivoted back to Deep Reinforcement Learning (PPO) using a much stronger **Lopunny/Froslass Control** deck.

**Why RL Succeeded Here:**
Instead of brute-forcing the learning from scratch, we used **Behavioral Cloning (BC)** to initialize the PPO agent's weights using our best heuristics, and implemented a **Pure-NumPy Inference Pipeline** to completely bypass the PyTorch `kaggle_environments` inference timeouts and crashes. 
By training on a diverse **Meta Opponent Curriculum** across 40,000+ episodes (with a critical memory leak fix preventing VRAM exhaustion), the Deep RL agent successfully learned advanced stalling and gusting strategies that the heuristic agent was incapable of executing, achieving a solid 282.3 Elo in Phase 9 and currently training to break the 303.6 peak in Phase 10.

---

## Phase 12: Mega Lucario Rule-Based (CURRENT)

The project ultimately abandoned PPO after its 342.0 peak and moved to the Mega Lucario list in `deck.csv`. Its deterministic, deck-specific heuristic reached **958.8 on the real ladder with v19**, far above every RL build.

The current v25 candidate keeps this deck unchanged. v24's 32-game ladder batch averaged roughly 600 and confirmed that the remaining losses were primarily policy errors, not missing deck pieces: incorrect prize perspective, premature bench-target attacks before gusting, search-plan leakage, unsafe forced promotions, unmodeled visible backup attackers, and misrouted lethal or retreat-enabling attachments. v25 corrects those decisions and verifies spread attacks are actually reachable. Replay 90715746 showed that the Alakazam rules also needed to preserve a safe Mega before hand resets and prevent chained Hariyama gusts from undoing a defensive stall. Keeping the list fixed isolates policy quality against the v19 promotion gate.
