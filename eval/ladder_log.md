# Kaggle Ladder Evaluation Log

| Date | Agent Version | Deck | Ladder Elo / μ | Local Win Rate vs Baseline | Notes |
|---|---|---|---|---|---|
| 2026-07-31 | Phase 1 (Rule-based) | Mono-Water | ERROR | N/A | Crashed: returned `[]` on Turn 0 instead of deck. |
| 2026-07-31 | Phase 1 v2 (Rule-based) | Mono-Water | ERROR | N/A | Crashed: illegal dummy deck (8x Cutiefly). |
| 2026-07-31 | Phase 1 v3 (Rule-based) | Mono-Water (Legal) | **170.1** | N/A | First working submission. Legal deck. Very passive heuristic. |
| 2026-07-31 | Phase 2 RL Baseline | Mono-Water (Legal) | ERROR | N/A | Crashed: Turn 0 returned `[]` (deck fix missing in policy.py). |
| 2026-07-31 | Phase 2 RL (Turn 0 Fix) | Mono-Water (Legal) | ERROR | N/A | Crashed: PyTorch import assumed; container has it but unknown crash. |
| 2026-07-31 | Phase 2 RL (Pure Numpy Fix) | Mono-Water (Legal) | **303.6** | Lost 1 local match | Active. Numpy inference, Turn 0 fixed. Slight improvement over Phase 1. |
| 2026-08-01 | Phase 3 True PPO (78k eps) | Snorlax (4 Pokémon) | **154.6** | N/A | Bench-out regression. Deck upgrade mandatory. |
| 2026-08-01 | Phase 4 v1 Bellibolt | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | Missing `__init__.py` — `ImportError` on startup. |
| 2026-08-01 | Phase 4 v2 Bellibolt | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | 2x Master Ball illegal — ACE SPEC max 1 per deck. |
| 2026-08-01 | Phase 4 v3 Bellibolt | Iono's Bellibolt / Kilowattrel | **PENDING** | N/A | Fix: 1x Master Ball + 1x Love Ball. Crustle detection, cooldown, Kilowattrel pivot. |
| 2026-08-01 | Phase 4 v4 Bellibolt (ctx fix) | Iono's Bellibolt / Kilowattrel | **PENDING** | N/A | **Key fix**: `select.context` is an INTEGER in real engine (e.g. 8=bench). Old string matching caused INVALID at step 21. Added `_handle_card_select()` + debug log. |
