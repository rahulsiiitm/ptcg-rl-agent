# Pokémon TCG AI Battle Challenge Agent

RL/heuristic agent for the Kaggle **"Pokémon TCG AI Battle Challenge - Simulation"** competition, running on the `cabt` C++ engine.

- **Entry Deadline**: Aug 9, 2026 · **Final Deadline**: Aug 16, 2026
- **Engine Docs**: https://matsuoinstitute.github.io/cabt/
- **Current Phase**: Phase 2 complete — RL pipeline deployed. Phase 3 (at-scale self-play) in preparation.

---

## Project Structure

```
ptcg-rl-agent/
├── main.py                  # Thin Kaggle entrypoint (loads policy_agent)
├── deck.csv                 # 60-card deck submitted with the agent
├── src/
│   ├── agent/
│   │   ├── rule_based.py    # Phase 1: Heuristic rule-based agent
│   │   ├── policy.py        # Phase 2: Pure-NumPy RL policy inference
│   │   ├── state_encoder.py # Encodes cabt JSON obs → flat float32 vector
│   │   ├── action_mask.py   # Masks illegal actions from logits (NumPy)
│   │   └── opponent_model.py# Probabilistic opponent hand/deck tracker
│   ├── env/
│   │   ├── fast_sim.py      # Gymnasium wrapper around the C++ cg engine
│   │   └── reward.py        # Shaped reward (win/loss + prizes + KOs + board)
│   └── train/
│       ├── train_ppo.py     # PPO training loop (PyTorch + CUDA)
│       └── self_play.py     # Checkpoint pool for self-play training
├── models/
│   ├── ppo_baseline.pth     # PyTorch weights (local training only)
│   └── ppo_weights.npz      # NumPy exported weights (submitted to Kaggle)
├── decks/                   # Deck rationale documents
├── eval/
│   ├── ladder_log.md        # Real ladder submission log
│   └── local_vs_ladder.md   # Local sim vs ladder comparison
└── scripts/                 # Build, validate, submit helpers
```

---

## Setup

1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dataset**:
   Authenticate with Kaggle (`~/.kaggle/kaggle.json`), then download:
   ```bash
   python -c "import kagglehub; kagglehub.competition_download('pokemon-tcg-ai-battle-challenge-strategy')"
   ```
   Place `EN_Card_Data.csv` in the `data/` folder.

---

## Usage

**Run a Local Test Match**
```bash
.venv\Scripts\python tests\test_agent.py
```
Runs a full match between `policy_agent` (P0) and `rule_based_agent` (P1) using the local C++ sim.

**Train the PPO Model** (uses RTX 3050 CUDA)
```bash
.venv\Scripts\python src\train\train_ppo.py
```
Saves weights to `models/ppo_baseline.pth` and exports `models/ppo_weights.npz` for submission.

**Build & Submit to Kaggle**
```bash
tar -czvf submission.tar.gz main.py deck.csv src models
kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m "message"
```

---

## Project Status & Phase Log

| Phase | Status | Description |
|---|---|---|
| **Phase 0** — Deck Selection | ✅ Complete | Mono-Water deck (4x Squirtle, 4x Staryu, 4x Poliwag, 48x Water Energy). Strictly legal 60-card deck. |
| **Phase 1** — Rule-Based Agent | ✅ Complete | `rule_based_agent`. Heuristic: Attack > Evolve > Play > Attach > End. First working submission scored **170.1**. |
| **Phase 2** — RL Pipeline | ✅ Deployed | PPO trained locally with CUDA. Pure-NumPy inference at Kaggle runtime. Scored **303.6** — current best. |
| **Phase 3** — At-Scale Self-Play | 🔄 Next | Multiprocessing parallel envs, self-play checkpoint pool, competitive deck upgrade. Target: beat 303.6. |

---

## Phase 2 Architecture Details

### Key Design Decision: Pure NumPy Inference (Phase 2)

> **Container fact**: The `cabt` environment is built `FROM gcr.io/kaggle-images/python:v163` —
> the **full Kaggle Python image** — so PyTorch, NumPy, and all standard ML libraries ARE available.
> The actual Phase 2 crashes were **Turn 0 deck-submission bugs** (returning `[]` instead of the
> 60-card list), not missing imports.

We nonetheless keep inference in pure NumPy for Phase 2 (faster cold-start, no CUDA spin-up). For Phase 3, `policy.py` can be upgraded to load and run the PyTorch model directly.

1. **Train locally** with PyTorch + CUDA (`train_ppo.py`) — full GPU-accelerated PPO.
2. **Export weights** to `models/ppo_weights.npz` (plain NumPy `.npz` format) for Phase 2 submission.
3. **Phase 3+**: Load `ppo_baseline.pth` directly with `torch.load()` — fully supported in the container.

### RL Pipeline Components

| Component | File | Description |
|---|---|---|
| Environment | `src/env/fast_sim.py` | Gymnasium wrapper; RL agent is P0, `rule_based_agent` auto-drives P1 |
| State Encoder | `src/agent/state_encoder.py` | 24-dim vector: 4 global + 10 self + 10 opponent features |
| Action Mask | `src/agent/action_mask.py` | Zeros out logits for options beyond `len(options)` |
| Reward | `src/env/reward.py` | Shaped: win/loss ± prizes taken/given |
| Actor-Critic | `src/train/train_ppo.py` | MLP(24→128→128→150), trained with 1-step TD PPO |
| Policy (Kaggle) | `src/agent/policy.py` | Pure NumPy forward pass; loads `ppo_weights.npz` |

### Known Limitations (to address in Phase 3)

- State vector is only **24-dimensional** — missing active Pokémon HP, type, energy attached, move names.
- PPO trained for only **~50 episodes** — proof-of-concept scale, not converged.
- **No self-play** — policy only ever faces the rule-based opponent during training.
- **Deck is weak** — Mono-basics with no Trainer cards (no draw, no switching).
- Phase 3 can safely use PyTorch directly at inference since the full `gcr.io/kaggle-images/python:v163` image is the container base.

---

## Strategy

See `AGENTS.md` for build order and architectural conventions.
See `decks/deck_rationale.md` for deck selection reasoning.
See `eval/ladder_log.md` for the full real-ladder submission log.
