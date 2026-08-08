# v25 Replay Audit (2026-08-08)

## Scope

- 40 unique Kaggle episodes, IDs 90715746 through 90925699.
- Result: **19 wins, 21 losses (47.5%)**; reported ladder average roughly 600.
- Pure Alakazam: **1-8**. Excluding pure Alakazam, v25 went **18-13 (58.1%)**.
- The historical v19 audit was also weak into Alakazam (1-13), so v19's
  958.8 peak did not represent a solved matchup.

## Confirmed causes

1. **Powerful Hand audit error (corrected in v27)**: the initial audit wrongly
   claimed the attack counts the defending Lucario player's hand. The card text
   says "your hand," meaning the attacking Alakazam player's hand. v25's
   perspective was correct; v26 reversed it and regressed the matchup model.
2. **Terminal attack tie**: in 90730426, Aura Jab and the planned game-winning
   Mega Brave both received 200000. Legal-option order selected nonlethal Aura
   Jab and lost the game.
3. **Over-restricted Mega evolution**: five of eight pure-Alakazam losses never
   evolved Mega Lucario. The fixed two-Energy gate denied safe, necessary
   one-Energy evolutions.
4. **Deck-out/stall**: 90723054, 90801870, and 90836708 reached zero deck after
   optional thinning or after a ready attacker was switched away / an
   uncharged Mega was left vulnerable to Boss stall.
5. **Energy over-specialization**: after Hariyama was already ready against
   Crustle, v25 still charged a second Makuhita rather than storing one Energy
   on Mega Lucario. In 90836708 that missing Energy prevented the final Mega
   Brave after Boss's Orders.

## v15 comparison

The checked-in `v15/` artifact is an Alakazam/Dudunsparce agent, not an older
Lucario policy. Its transferable strengths are strict safe-draw accounting,
resource-aware evolution/retreat logic, a legal fallback, and a search layer
that overrides the heuristic only by a meaningful margin. v26 adopts the
safe-draw discipline while keeping the proven Lucario deck and policy core.

## v26 changes (historical; Alakazam model superseded)

- Powerful Hand was incorrectly changed to our hand size; v27 reverts this.
- Carmine/Lillie were incorrectly suppressed against Alakazam; v27 removes
  that suppression because our hand does not power the opponent's attack.
- Safe one-Energy Mega evolutions are allowed; unsafe three-prize evolutions
  remain blocked.
- Only the attack matching a terminal plan gets terminal priority.
- Low-deck search gating is unconditional and activates at the safety boundary.
- A ready Active attacker receives low-deck tempo priority over a Switch that
  can open a Boss stall.
- Once Hariyama is ready, Crustle and Alakazam energy routing diversifies into
  Lucario instead of overbuilding a redundant second Hariyama.

## Validation gate

Local simulation remains a sanity check, not promotion evidence. v26 must be
tested on the real ladder and is not promoted over v19 until it beats 958.8.

## Real-cabt matchup sanity check

`scripts/run_realistic_gauntlet.py` runs both actual submission policies in the
real bundled cabt engine, alternates seats, and preserves the last 20 decisions
from every candidate loss. In the expanded run against the checked-in v15
Alakazam/Dudunsparce submission, v26 scored **11-29 over 40 games (27.5%)** with
bounded search enabled: 7-13 as seat 0 and 4-16 as seat 1. An earlier 12-game
searched sample was 5-7, making the combined searched evidence 16-36 over 52
games. The same heuristic with candidate search disabled scored **1-11 over 12
games**. These are separate randomized samples rather than paired seeded games,
but search remains clearly preferable to the no-search policy.

The expanded run contained no crash, illegal selection, or candidate deck-out
loss. Alakazam took all prizes in 27 losses; two more were early lone-active
knockouts before Lucario established a Bench. In the 27 ordinary losses, v26
still had at least four prizes remaining 16 times, showing that the dominant
failure is setup/attack tempo rather than only a narrow final-turn mistake.
The result confirms that v26 does not solve Alakazam. These simulations were
later understood to include the reversed Powerful Hand model.

### Rejected Alakazam experiments

Three matchup-gated changes were isolated and tested in additional real-cabt
samples. None passed the promotion gate, so all were reverted from the agent:

- exact "safe" Carmine/Lillie expansion: 7-13 in the 20-game gate, then 10-30
  in the 40-game confirmation;
- early two-Energy Mega routing with strict hand suppression: 5-15;
- Hero Cape priority on Hariyama: 2-18.

The combined safe-draw plus fast-Mega experiment was also 5-15. These results
showed no reliable improvement and were reverted.

## v27 correction and deck experiment

The official card data confirms that Powerful Hand uses the attacking
Alakazam player's hand. After correcting the threat model and restoring normal
Carmine/Lillie setup, v27 replaced four Dusk Balls with four Hand Trimmers.
The policy plays them only when visible Alakazam has more than five cards or
the Mega Froslass axis threatens us while our hand exceeds five.

- Corrected original deck gate: 4-16 against v15 Alakazam.
- Hand Trimmer gate: 10-10.
- Hand Trimmer confirmation: 20-20.
- Combined Hand Trimmer evidence: 30-30 over 60 games.
- Hand Trimmer list versus original Lucario list: 19-21 over 40 games.
- No crash or illegal action; all 43 replay regressions pass in the final v27 build.

This is local sanity evidence rather than a ladder guarantee, but Hand Trimmer
is the first experiment to produce a large, repeatable Alakazam improvement
without a meaningful Lucario-mirror collapse.
