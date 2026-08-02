<h1 align="center"><img width="1024" height="384" alt="pokemongames-banner-1024x384" src="https://github.com/user-attachments/assets/7be6b89f-fd42-49a8-af15-7140c7d72c22" />
PTCG AI Battle Agent</h1>

<p align="center">
  <strong>Reinforcement Learning + Heuristic Agent for the Kaggle Pokémon TCG AI Battle Challenge</strong>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/pokemon-tcg-ai-battle">
    <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/>
  </a>
  <a href="https://matsuoinstitute.github.io/cabt/">
    <img src="https://img.shields.io/badge/Engine-cabt%20C%2B%2B-FF6B35?style=for-the-badge&logo=cplusplus&logoColor=white" alt="cabt engine"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-PPO%20Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://gymnasium.farama.org/">
    <img src="https://img.shields.io/badge/Gymnasium-RL%20Env-3CBAC0?style=for-the-badge" alt="Gymnasium"/>
  </a>
  <img src="https://img.shields.io/badge/Phase-3%20Scaling-22C55E?style=for-the-badge" alt="Phase 3"/>
  <img src="https://img.shields.io/badge/Best%20Score-303.6-FFD700?style=for-the-badge&logo=star&logoColor=black" alt="Score"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Entry%20Deadline-Aug%209%202026-FF4444?style=flat-square"/>
  <img src="https://img.shields.io/badge/Final%20Deadline-Aug%2016%202026-FF4444?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-RTX%203050-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Competition%20Terms-gray?style=flat-square"/>
</p>

---

## Overview

A competitive **PPO Reinforcement Learning** agent for the Kaggle Pokémon TCG AI Battle Challenge. The agent is trained locally using PyTorch with CUDA acceleration and deployed to Kaggle using a pure-NumPy inference pipeline for maximum runtime compatibility.

The project follows a strict **phase-gated deployment cycle**: every new agent iteration must empirically outperform the previous production version on the real Kaggle ladder before being promoted to the `main.py` entrypoint.

| Metric | Value |
|---|---|
| Best Ladder Score | **303.6** (Phase 2 RL) |
| Submission Format | `submission.tar.gz` (main.py + deck.csv + src + models) |
| Time Budget | 600s per match |
| Deck Configuration | Mono-Water Basic (60 cards) |
| Policy Architecture | PPO Actor-Critic (MLP 24→128→128→150) |
| Training Hardware | AMD Ryzen 5 5600H + RTX 3050 (CUDA) |

---

## Project Structure

```
ptcg-rl-agent/
├── main.py                  # Thin Kaggle entrypoint (loads policy_agent)
├── deck.csv                 # 60-card deck submitted with the agent
├── src/
│   ├── agent/
│   │   ├── rule_based.py         # Phase 1: Heuristic rule-based agent
│   │   ├── rule_based_generic.py # Universal "True Sight" opponent for all 24 meta decks
│   │   ├── policy.py             # Phase 2: Pure-NumPy RL policy inference
│   │   ├── state_encoder.py      # Encodes cabt JSON obs → flat float32 vector
│   │   ├── action_mask.py        # Masks illegal actions from logits (NumPy)
│   │   └── opponent_model.py     # Probabilistic opponent hand/deck tracker
│   ├── env/
│   │   ├── fast_sim.py         # Gymnasium wrapper around the C++ cg engine
│   │   └── reward.py           # Shaped reward (win/loss + prizes + KOs)
│   └── train/
│       ├── train_ppo.py        # PPO training loop (PyTorch + CUDA)
│       └── self_play.py        # Checkpoint pool for self-play training
├── models/
│   ├── ppo_baseline.pth        # PyTorch checkpoint (local training only)
│   └── ppo_weights.npz         # NumPy exported weights (Kaggle inference)
├── assets/
│   └── banner.png              # Project banner
├── decks/                   # Deck rationale and meta configurations
├── eval/
│   ├── ladder_log.md           # Real ladder submission log
│   └── local_vs_ladder.md      # Local sim vs. ladder comparison
└── scripts/                 # Build, validate, and submission helpers
```

---

## Setup and Installation

### 1. Virtual Environment Initialization

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Dataset Acquisition

Authenticate with Kaggle (`~/.kaggle/kaggle.json`), then download the card dataset:

```bash
python -c "import kagglehub; kagglehub.competition_download('pokemon-tcg-ai-battle-challenge-strategy')"
```

Place `EN_Card_Data.csv` in the `data/` folder.

---

## Usage

### Run a Local Test Match

```bash
.venv\Scripts\python tests\test_agent.py
```

Executes a full match between `policy_agent` (Player 0) and `rule_based_agent` (Player 1) using the local C++ simulation engine.

### Train the PPO Model

```bash
.venv\Scripts\python src\train\train_ppo.py
```

Initiates the PyTorch training loop on CUDA. Saves weights to `models/ppo_baseline.pth` and `models/ppo_weights.npz`.

### Build and Submit to Kaggle

```bash
tar -czvf submission.tar.gz main.py deck.csv src models
kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m "your message"
```

> **Note on Rate Limits:** The Kaggle API enforces a strict daily limit of 5 submissions. Always validate the agent locally before submission.

---

## Phase Status and Ladder Results

| Phase | Status | Real Ladder Score | Notes |
|---|---|---|---|
| **Phase 0** — Deck Selection | Complete | — | Mono-Water: 4x Squirtle, 4x Staryu, 4x Poliwag, 48x Water Energy |
| **Phase 1** — Rule-Based Agent | Complete | **170.1** | Heuristic: Attack > Evolve > Play > Attach > End |
| **Phase 2** — RL Pipeline | Deployed | **303.6** | PPO + Pure-NumPy inference. Current active deployment. |
| **Phase 4** — Bellibolt Heuristic | Complete | **157.6** | Rule-based agent with Iono's Bellibolt deck. |
| **Phase 9** — RL Agent (BC Init) | Complete | **282.3** | PPO initialized from Behavioral Cloning on Lopunny deck. |
| **Current** — Meta Opponent Curriculum | In Progress | Target: **>303.6** | PPO trained against all meta decks to generalize strategy. |

Full submission history → [`eval/ladder_log.md`](eval/ladder_log.md)

---

## Phase 2 Architecture

### Key Design Decisions

> **Environment Context**: The `cabt` environment is built `FROM gcr.io/kaggle-images/python:v163` — the full Kaggle Python image. **PyTorch IS available** at the Kaggle runtime. Phase 2 crashes were identified as Turn 0 deck-submission exceptions, not import resolution errors. Phase 3 can safely `import torch` directly in `policy.py`.

We utilize **pure NumPy inference** in Phase 2 to ensure a fast cold-start, training with PyTorch locally and exporting the network weights to `.npz`:

```
┌──────────────────────┐     export     ┌────────────────────────┐
│  train_ppo.py        │ ─────npz──────▶│  policy.py (Kaggle)    │
│  PyTorch + CUDA      │                │  Pure NumPy forward    │
│  RTX 3050 (local)    │                │  relu(W·x + b) × 3     │
└──────────────────────┘                └────────────────────────┘
```

### RL Pipeline Components

| Component | File | Description |
|---|---|---|
| Environment | `src/env/fast_sim.py` | Gymnasium wrapper; RL agent is P0, rule-based auto-drives P1 |
| State Encoder | `src/agent/state_encoder.py` | 24-dim: 4 global + 10 self + 10 opponent |
| Action Mask | `src/agent/action_mask.py` | NumPy masked softmax over variable-length legal moves |
| Reward | `src/env/reward.py` | Shaped: win/loss terminal + prize delta per step |
| Actor-Critic | `src/train/train_ppo.py` | MLP(24→128→128→150), 1-step TD PPO |
| Policy (Kaggle) | `src/agent/policy.py` | Pure NumPy forward pass; loads `ppo_weights.npz` |

### Known Limitations & Phase 3 Roadmap

| Limitation | Phase 3 Solution |
|---|---|
| 24-dim state (no HP, type, moves) | Expand to ~128-dim encoder |
| Only 50 training episodes | 100k+ episodes with parallel envs |
| No self-play | Checkpoint pool self-play |
| Weak mono-basic deck | Snorlax control deck |

---

## Key Development Rules

- **Execution Safety** — The agent must never crash. It must always return a legal fallback action. Crashes result in an instant loss.
- **Turn 0 Initialization** — Turn 0 must always return the deck configuration (60 card IDs). All subsequent turns return action indices.
- **Deployment Protocol** — Never promote a Phase N+1 agent to `main.py` unless it empirically beats the previous agent's real ladder score.
- **Ladder Authority** — Local simulations differ from the real ladder. Local results are sanity checks only; the real ladder is the sole source of ground truth.

---

## References

- [cabt Engine Documentation](https://matsuoinstitute.github.io/cabt/)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
- [Ladder Log](eval/ladder_log.md) · [AGENTS.md](AGENTS.md) · [Deck Rationale](decks/deck_rationale.md)
