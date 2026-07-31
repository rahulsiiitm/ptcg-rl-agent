# Kaggle Ladder Evaluation Log

| Date | Agent Version | Deck | Ladder Elo / μ | Local Win Rate vs Baseline | Notes |
|---|---|---|---|---|---|
| 2026-07-31 | Phase 1 (Rule-based) | Mono-Water | ERROR | N/A | Initial robust heuristic agent. Crashed due to returning `[]` on turn 0. |
| 2026-07-31 | Phase 1 v2 (Rule-based) | Mono-Water | ERROR | N/A | Bugfix: returned deck. Crashed due to illegal dummy deck (8x Cutiefly). |
| 2026-07-31 | Phase 1 v3 (Rule-based) | Mono-Water (Legal) | 300.0 | N/A | Bugfix: built strictly legal Mono-Water deck. Very weak, loses most games. |
| 2026-07-31 | Phase 2 (Numpy Baseline) | Mono-Water (Legal) | TBD | N/A | Rewrote policy to pure NumPy. Fixed PyTorch Kaggle container crash. |
