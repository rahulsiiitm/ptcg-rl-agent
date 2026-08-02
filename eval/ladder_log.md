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
| 2026-08-01 | Phase 4 v3 Bellibolt | Iono's Bellibolt / Kilowattrel | **157.6** | N/A | Fix: 1x Master Ball + 1x Love Ball. Survived! |
| 2026-08-01 | Phase 4 v4 Bellibolt (ctx fix) | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | Still errored out. |
| 2026-08-01 | Hybrid RL Agent (500 ep test) | Lopunny/Froslass | **224.9** | N/A | After main.py fixes, successfully scored 224.9! |
| 2026-08-02 | Phase 6 PPO Lookahead | Lopunny/Froslass | **ERROR** | N/A | Crashed on Kaggle. |
| 2026-08-02 | Phase 7 RL Agent (Fallback fix) | Lopunny/Froslass | **186.0** | N/A | Survived with minCount fix, but performance dropped slightly. |
| 2026-08-02 | Phase 1 Heuristic Pivot | Lopunny/Froslass | **51.4** | N/A | Very low score for pure heuristic on this deck. |
| 2026-08-02 | Phase 8 Heuristic | Lopunny/Froslass | **ERROR** | N/A | Errored out. |
| 2026-08-02 | Phase 9 RL Agent (BC init) | Lopunny/Froslass | **282.3** | N/A | Great score! RL agent initialized from BC, getting very close to the 303.6 peak! |
| 2026-08-02 | Phase 10 NumPy Pipeline | Lopunny/Froslass | **Pending** | N/A | PPO agent trained on diverse curriculum. Fixed CUDA VRAM memory leak in training loop! |
