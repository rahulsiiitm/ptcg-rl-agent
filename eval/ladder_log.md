# Kaggle Ladder Evaluation Log

| Date | File Name | Agent Version | Deck | Ladder Elo / μ | Local Win Rate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-31 | `submission.tar.gz` | Phase 1 (Rule-based) | Mono-Water | **ERROR** | N/A | Crashed: returned `[]` on Turn 0 instead of deck. |
| 2026-07-31 | `submission.tar.gz` | Phase 1 v2 (Rule-based) | Mono-Water | **ERROR** | N/A | Crashed: illegal dummy deck (8x Cutiefly). |
| 2026-07-31 | `submission.tar.gz` | Phase 1 v3 (Rule-based) | Mono-Water (Legal) | **170.1** | N/A | First working submission. Legal deck. Very passive heuristic. |
| 2026-07-31 | `submission.tar.gz` | Phase 2 RL Baseline | Mono-Water (Legal) | **ERROR** | N/A | Crashed: Turn 0 returned `[]` (deck fix missing in policy.py). |
| 2026-07-31 | `submission.tar.gz` | Phase 2 RL (Turn 0 Fix) | Mono-Water (Legal) | **ERROR** | N/A | Crashed: PyTorch import assumed; container has it but unknown crash. |
| 2026-07-31 | `submission.tar.gz` | Phase 2 RL (Pure Numpy Fix) | Mono-Water (Legal) | **303.6** | Lost 1 local match | Active. Numpy inference, Turn 0 fixed. Slight improvement over Phase 1. |
| 2026-08-01 | `submission.tar.gz` | Phase 3 True PPO (78k eps) | Snorlax (4 Pokémon) | **154.6** | N/A | Bench-out regression. Deck upgrade mandatory. |
| 2026-08-01 | `submission.tar.gz` | Phase 4 v1 Bellibolt | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | Missing `__init__.py` — `ImportError` on startup. |
| 2026-08-01 | `submission.tar.gz` | Phase 4 v2 Bellibolt | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | 2x Master Ball illegal — ACE SPEC max 1 per deck. |
| 2026-08-01 | `submission.tar.gz` | Phase 4 v3 Bellibolt | Iono's Bellibolt / Kilowattrel | **157.6** | N/A | Fix: 1x Master Ball + 1x Love Ball. Survived! |
| 2026-08-01 | `submission.tar.gz` | Phase 4 v4 Bellibolt (ctx fix) | Iono's Bellibolt / Kilowattrel | **ERROR** | N/A | Still errored out. |
| 2026-08-01 | `submission.tar.gz` | Hybrid RL Agent (500 ep test) | Lopunny/Froslass | **224.9** | N/A | After main.py fixes, successfully scored 224.9! |
| 2026-08-02 | `submission.tar.gz` | Phase 6 PPO Lookahead | Lopunny/Froslass | **ERROR** | N/A | Crashed on Kaggle. |
| 2026-08-02 | `submission.tar.gz` | Phase 7 RL Agent (Fallback fix) | Lopunny/Froslass | **186.0** | N/A | Survived with minCount fix, but performance dropped slightly. |
| 2026-08-02 | `submission.tar.gz` | Phase 1 Heuristic Pivot | Lopunny/Froslass | **51.4** | N/A | Very low score for pure heuristic on this deck. |
| 2026-08-02 | `submission.tar.gz` | Phase 8 Heuristic | Lopunny/Froslass | **ERROR** | N/A | Errored out. |
| 2026-08-02 | `submission.tar.gz` | Phase 9 RL Agent (BC init) | Lopunny/Froslass | **282.3** | N/A | Great score! RL agent initialized from BC, getting very close to the 303.6 peak! |
| 2026-08-02 | `submission.tar.gz` | Phase 10 NumPy Pipeline | Lopunny/Froslass | **342.0** | 60% (6/10) | Peaked at 600+ but settled at 342. Still a NEW ALL-TIME HIGH! Reached 55k eps. |
| 2026-08-03 | `submission.tar.gz` | Phase 11 PPO Ensemble (82k eps) | Lopunny/Froslass | **342.0** | 22-38% (50 games) | Submitted to Kaggle with memory leak fix and 3-model ensemble. |
| 2026-08-03 | `lucario_submission.tar.gz` | Rule-Based Mega Lucario ex | Mega Lucario | **ERROR** | N/A | Missing `cg` engine dependencies. |
| 2026-08-03 | `lucario_submission.tar.gz` | Rule-Based Mega Lucario ex | Mega Lucario | **615.4** | N/A | First working Mega Lucario submission with `cg` binaries and Turn 0 fallback. |
| 2026-08-04 | `submission.tar.gz` | Phase 12 Rule-Based Lucario | Mega Lucario | **167.3** | N/A | Added context-scoring for DISCARD/LOOK and deck-count throttles. |
| 2026-08-04 | `submission.tar.gz` | Mega Lucario (Stable API Patch) | Mega Lucario | **371.3** | N/A | Overhaul of heuristic property access to fix T1 engine crashes. |
| 2026-08-04 | `valid_submission.tar.gz` | Rule-Based Lucario (Enum Fix) | Mega Lucario | **ERROR** | N/A | Crashed on invalid Enum conversion. |
| 2026-08-04 | `valid_submission.tar.gz` | Rule-Based Lucario (Root Deck Fix) | Mega Lucario | **390.3** | N/A | Root `deck.csv` lookup path fix. |
| 2026-08-04 | `valid_submission.tar.gz` | Rule-Based Lucario (Safety Fixes) | Mega Lucario | **236.0** | N/A | Deck-Out and Aura Jab safety rules active. |
| 2026-08-04 | `v15.tar.gz` | Rule-Based Lucario v15 | Mega Lucario | **775.5** | N/A | Mega Lucario baseline peak before replay analysis. |
| 2026-08-05 | `submission.tar.gz` | Phase 12 Rule-Based Baseline | Mega Lucario | **691.3** | 76% (38/50) | Switched entirely to Mega Lucario heuristic. Abandoned RL. |
| 2026-08-06 | `submission.tar.gz` | Phase 12 Heuristic (Unpatched) | Mega Lucario | **420.3** | N/A | Interim submission before replay-based misplay fixes. |
| 2026-08-06 | `submission.tar.gz` | **Rule-Based Lucario (Replay-Patched)** | Mega Lucario | **902.4** (Peak **936.0**) | 94% (16/17) | Replay fixes for Attack Priority (+40k), Boss Orders (+47k), and Hero Cape (+8k). |
| 2026-08-06 | `v17.tar.gz` | **Rule-Based Lucario v17** | Mega Lucario | **774.2** | 50% (5/10) | **CRASHED ELO:** The +40k Attack boost caused the agent to attack instantly and skip playing setups. Supporter math was also flawed. |
| 2026-08-06 | `v18.tar.gz` | **Rule-Based Lucario v18 (Hotfix)** | Mega Lucario | *Pending Ladder Upload* | 100% | **v18 BUILD:** Removed the +40k attack bug and fixed the Supporter hand-size penalty math. Target: Recover to 1000+ Elo. |



---

## Detailed Evaluation Notes

- **2026-08-04**: `rule_based_lucario.py` (Bugfix/Mock Objects). Deck: Mega Lucario. Local simulation validated 100% crash-free.
- **2026-08-06**: Replay-based misplay analyzer deployed on 23 Kaggle replays; identified and patched 3 core tactical errors.

---

## Ladder Milestones & Trajectory

- **Initial Baseline (Mono-Water Rule-Based)**: 170.1
- **RL Peak (Lopunny/Froslass PPO)**: 342.0 (Abandoned due to sub-optimal decisions under strict Kaggle CPU budget)
- **Phase 12 Heuristic Baseline (Mega Lucario)**: 615.4 → 691.3 → 775.5
- **Phase 12 Replay-Patched Peak**: **936.0 Peak** / **902.4 Active Score** (Attack priority +40k, Boss Orders +47k, Hero Cape +8k)
