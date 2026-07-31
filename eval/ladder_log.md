# Kaggle Ladder Evaluation Log

| Date | Agent Version | Deck | Ladder Elo / μ | Local Win Rate vs Baseline | Notes |
|---|---|---|---|---|---|
| 2026-07-31 | Phase 1 (Rule-based) | Mono-Water | ERROR | N/A | Crashed: returned `[]` on Turn 0 instead of deck. |
| 2026-07-31 | Phase 1 v2 (Rule-based) | Mono-Water | ERROR | N/A | Crashed: illegal dummy deck (8x Cutiefly). |
| 2026-07-31 | Phase 1 v3 (Rule-based) | Mono-Water (Legal) | **170.1** | N/A | First working submission. Legal deck. Very passive heuristic. |
| 2026-07-31 | Phase 2 RL Baseline | Mono-Water (Legal) | ERROR | N/A | Crashed: Turn 0 returned `[]` (deck fix missing in policy.py). |
| 2026-07-31 | Phase 2 RL (Turn 0 Fix) | Mono-Water (Legal) | ERROR | N/A | Crashed: PyTorch import assumed; container has it but unknown crash. |
| 2026-07-31 | Phase 2 RL (Pure Numpy Fix) | Mono-Water (Legal) | **303.6** | Lost 1 local match | Active. Numpy inference, Turn 0 fixed. Slight improvement over Phase 1. |

