# v27 Endgame Meta Audit (2026-08-08)

## Replay population

All 41 JSON files currently available in `Downloads` were inspected. After
deduplicating by Kaggle EpisodeId and rejecting incomplete files, 40 unique
episodes remained. The five archetypes named as the endgame field represented
30/40 games (75%):

| Archetype | Games | v25 record |
|---|---:|---:|
| Alakazam | 9 | 1-8 |
| Mega Lucario | 7 | 6-1 |
| Starmie/Froslass, including Grimmsnarl hybrid | 7 | 5-2 |
| Dwebble/Crustle | 4 | 1-3 |
| Archaludon | 3 | 1-2 |

The older replay analyzer labeled card 190 as Typhlosion. Official card data
identifies 169/190 as Duraludon/Archaludon; the analyzer is corrected in v27.

## Rules correction

Alakazam's Powerful Hand says to place two counters for each card in "your
hand." That is the attacking Alakazam player's hand. v26 incorrectly changed
the threat model to Lucario's hand and suppressed Lucario's own draw
supporters. v27 restores the attacker-hand interpretation and normal setup.

## Deck decision

v27 replaces four Dusk Balls with four Hand Trimmers. Fighting Gong and Poke
Pad retain eight search outs for Basics and non-Rule-Box Pokemon. Hand Trimmer
is scored only after a visible Alakazam or Froslass axis makes it relevant:

- Alakazam: reduce a frequently 19-25-card attacking hand to five, capping
  Powerful Hand at 100 damage until the opponent rebuilds.
- Froslass: reduce Lucario's hand to five, capping Resentful Refrain at 250.
- Crustle: removal of Mega-only Dusk Ball does not remove the Hariyama/Solrock
  non-ex routes searched by Poke Pad and Fighting Gong.
- Archaludon: the explicit 220-damage threat model remains. Archaludon is not
  Fighting-weak, so this is a tempo matchup rather than a type-advantage one.
- Lucario: no matchup-specific Hand Trimmer activation.

## Real-cabt evidence

- Hand Trimmer list versus actual v15 Alakazam: 10-10 gate and 20-20
  confirmation, **30-30 over 60 games**.
- Hand Trimmer list versus original Lucario list: **19-21 over 40 games**.
- A delayed Hand Trimmer sequencing experiment went 28-31-1 across 60 and was
  rejected; v27 uses the measured 30-30 high-priority behavior.
- No crash or illegal selection occurred in these gauntlets.
- 45 replay regression tests pass.

## v26 ladder follow-up (15 unique games)

The 90952878-90963811 batch deduplicates to 15 games and a 7-8 record:

| Archetype | Record |
|---|---:|
| Mega Lucario | 3-1 |
| Archaludon | 1-2 |
| Dragapult | 1-2 |
| Team Rocket's Mewtwo | 1-1 |
| Other | 1-0 |
| Crustle | 0-1 |
| Alakazam | 0-1 |

The analyzer previously called card IDs 400/401 Gholdengo; the official data
identifies that replay as Team Rocket's Mewtwo/Spidops. Two actionable policy
faults appeared in the Archaludon losses: the agent repeatedly put manual
Energy on an already-ready, damaged bench Mega (four to eight Energy), and the
terminal-defense guard rejected an available attack when no safe pivot
existed. v27 now penalizes manual Energy past each attacker's useful ceiling
and permits an attack-enabling Active attachment as a last-chance line only
when no safe pivot exists. The Crustle loss also spent late Dusk Balls, which
the v27 deck no longer contains; the Alakazam loss is the exact large-hand case
targeted by Hand Trimmer.

The remaining losses include severe setup/energy bricks and positions where
Dragapult or Mewtwo already had the final-prize attack established. These
cannot honestly be converted into guaranteed wins by a heuristic change.

Only Alakazam and Lucario have authentic independent local policies. Crustle,
Archaludon, and Starmie/Froslass coverage is card-text/replay-regression based;
their real ladder performance remains unproven. Local cabt is a sanity check,
not a ladder predictor.
