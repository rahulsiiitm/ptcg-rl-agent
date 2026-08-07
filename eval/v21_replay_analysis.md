# v21 Replay Regression Analysis

Date: 2026-08-07  
Replay set: episodes 90605503–90621445 supplied from the real Kaggle ladder

## Result

- 23 unique replays: 11 wins, 12 losses (47.8% observed win rate).
- Episode 90609477 was supplied twice and is counted once.
- v19 remains the verified ladder peak at 958.8. No later version is promoted until it beats that result on the real ladder.

## Confirmed v21 regression

v21 compared a raw option value of `11` as if it meant End Turn. In cabt, `OptionType.DISCARD == 11`; End Turn is `OptionType.END == 14`. It also assigned `-1000` to ordinary Yes choices in common states. Those mistakes damaged Trainer and ability resolution and explain the sharp v21 ladder decline. v22 removed the raw numeric check and restored enum-based Yes/No handling.

## Replay-analysis bug found

The previous analyzers decoded actions from `visualize` search branches and paired step N's action with step N's observation. Both are incorrect:

- `visualize` contains internal lookahead branches, not only real moves.
- Kaggle stores an action on step N for the observation from step N-1.

This produced impossible reports such as End Turn followed by another action in the same turn. `scripts/analyze_replays_v3.ps1` now uses only real top-level actions, aligns them with the previous observation, and deduplicates episode IDs.

## New confirmed tactical mistake

Lunar Cycle has a hidden ordering cost: it discards a Fighting Energy from hand to draw three cards. In losses 90611080 and 90612665, the agent used Lunar Cycle while an undercharged attacker and a legal manual attachment were available. The only Energy disappeared before the once-per-turn attachment. The same pattern appeared later in 90621445.

v22 therefore enforces this invariant:

> If Lunar Cycle and a useful Fighting Energy attachment are both legal, attach to an undercharged attacking line first. Search may not override that ordering.

The guard applies only below useful energy goals (Lucario/Riolu 2, Hariyama/Makuhita 3, Solrock 1). If all attackers are charged, Lunar Cycle may still spend the Energy.

## Rejected false positive

Episode 90621372 showed an opposing bench Solrock at 10 HP, but the opposing Active Mega Lucario ex was at 40 HP. Boss's Orders would trade a three-Prize Active knockout for a one-Prize bench knockout, so this was not a missed Boss play. The old “bench below 40% HP means force Boss” rule is unsafe; target value and reachable knockout matter more than HP percentage alone.

## Validation policy

Local simulation remains a crash/sanity check only. v22 must be evaluated on the real ladder and compared directly with v19 before promotion.
