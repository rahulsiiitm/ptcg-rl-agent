# v19 Replay Audit (2026-08-07)

## Scope and method

- Received 59 JSON files covering episode IDs 90398158 through 90587053.
- Deduplicated by Kaggle `EpisodeId`: `90506199` and `90528500` each had a second identical download.
- Rejected `90422512-1.json` because it is an incomplete 3.8 KB payload without a replay.
- Final sample: **56 unique valid games: 26 wins, 30 losses (46.4%)**.
- Actions were decoded against the preceding observation because Kaggle records the action in step N for the selection presented in step N-1.
- Only real top-level actions were analyzed; internal search `visualize` branches were excluded.

The 958.8 rating remains v19's verified peak. This replay batch is a failure-analysis sample, not a replacement estimate of its ladder strength.

## Matchup results

| Visible opponent archetype | Wins | Losses | Win rate |
|---|---:|---:|---:|
| Alakazam | 1 | 13 | 7.1% |
| Grimmsnarl + Froslass | 6 | 2 | 75.0% |
| Mega Lucario | 4 | 2 | 66.7% |
| Crustle | 3 | 3 | 50.0% |
| Dragapult | 1 | 3 | 25.0% |
| Archaludon | 1 | 3 | 25.0% |
| Ogerpon / Hydrapple | 2 | 0 | 100% |
| Grimmsnarl without visible Froslass | 1 | 1 | 50.0% |
| Gholdengo-signature / Team Rocket | 1 | 1 | 50.0% |
| Other | 6 | 2 | 75.0% |

## Root causes

### 1. Alakazam was the dominant structural counter

Alakazam accounted for **13 of 30 losses (43.3%)**. `Powerful Hand` places two damage counters for every card in the opponent's hand, bypassing ordinary damage reduction and routinely removing a full-health Mega Lucario for three prizes. The Fighting resistance also turns Aura Jab's 130 base damage into 100 unless another effect modifies it.

- Across Alakazam losses, v19 selected Aura Jab 11 times while Mega Brave was also legal.
- Two of those choices are confirmed nonlethal Aura Jabs that left Alakazam at 10 HP (`90408802` and `90454390`).
- Several other games exposed an uncharged Mega on the Bench, allowing Boss's Orders plus Powerful Hand to take three prizes (`90424810`, `90443744`).
- The sole win (`90443021`) used Mega Brave to remove powered Alakazam and maintained multiple developed attackers.

### 2. One-Energy Dragapult was treated as harmless

The old policy checked the board as it existed during our turn, not after the opponent's normal Energy attachment. In `90404273`, v19 left Dragapult at 50 HP while it had one Energy; the opponent attached the second Energy and Phantom Dive took the damaged Mega plus Bench prizes. In `90406555`, active damage and six distributed counters created a multi-prize closing route.

### 3. Crustle required deliberate non-ex Energy routing

The Crustle split was 3-3. Its Ability prevents damage from Pokemon ex, yet losing games accumulated Energy on Mega Lucario while Hariyama and Solrock remained unable to attack. This was already addressed in v23 and is retained.

### 4. Archaludon damage was underestimated

Archaludon went 3-1 against v19. `Metal Defender` reaches 220 after the next manual attachment, while Relicanth exposes Duraludon's `Raging Hammer`, which scales with damage already on Archaludon. `90506199` also ended in a true Bench-out after Makuhita was the only Pokemon v19 established.

### 5. Remaining losses were mostly terminal prize sequencing

The Mega Lucario mirror, Grimmsnarl/Froslass, Mega Starmie, and Team Rocket Mewtwo losses commonly ended with a damaged multi-prize active or a one-prize Bench target available to Boss. v23's terminal prize-denial framework already covers the visible versions of these routes.

## v24 changes derived from this audit

- Add high target priority to Abra, Kadabra, and especially Alakazam.
- Prefer a modeled lethal Mega Brave over a nonlethal Aura Jab specifically against Alakazam; do not restore v22's global prize-greed bonus.
- Treat knocking out the visible terminal attacker as defense, so retreat logic cannot override the removal attack.
- Delay evolving Riolu into a three-prize Mega against Alakazam until Riolu has two Energy.
- Prioritize evolving and powering the single-prize Hariyama line in the Alakazam matchup.
- Include the opponent's next normal attachment when checking Dragapult, Ogerpon, Hydrapple, Mega Lucario, Froslass, Grimmsnarl, Crustle, Alakazam, and Archaludon threats.
- Model Archaludon's 220 damage and Relicanth-enabled Raging Hammer floor.

## Validation

- 14 focused replay-regression tests pass.
- Five Lucario mirror cabt smoke matches completed without crashes or illegal actions.
- Nine cross-deck cabt legality matches completed against Alakazam, Dragapult, and Crustle lists.
- The local Alakazam sanity run finished 3-0, but local results are not used as promotion evidence.
- v24 must beat v19's verified **958.8** on the real ladder before promotion.
