# Pokémon TCG AI Battle Challenge Agent

This repository contains the source code for an RL/Rule-Based agent built for the Kaggle "Pokémon TCG AI Battle Challenge - Simulation" competition.

## Project Structure

- `main.py`: The main entrypoint for the Kaggle engine. Currently loads the rule-based heuristic agent.
- `src/`: Agent logic (rule-based, RL wrappers, reward shaping).
- `data/`: Contains the dataset (downloaded separately) and parsing scripts (`card_lookup.py`).
- `decks/`: Deck building rationale and the final `deck.csv` submission file.
- `eval/`: Local HTML results and ladder evaluation logs.
- `scripts/`: Shell scripts for building, validating, and submitting the agent.

## Setup

1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dataset**:
   Ensure you are authenticated with Kaggle (`kaggle.json`), then download the dataset:
   ```bash
   python -c "import kagglehub; kagglehub.competition_download('pokemon-tcg-ai-battle-challenge-strategy')"
   ```
   Place the resulting files (`EN_Card_Data.csv`, etc.) in the `data/` folder.

## Usage

**Validate Locally**
Before submitting to Kaggle, ensure the agent does not crash by running a local self-play match:
```bash
./scripts/validate_submission.sh
```
This will output `eval/result.html` which you can open in a browser.

**Build for Kaggle**
```bash
./scripts/build_submission.sh
```
This generates `submission.tar.gz`.

**Submit**
```bash
./scripts/submit_to_kaggle.sh
```

## Strategy

Please see `AGENTS.md` and `decks/deck_rationale.md` for a detailed breakdown of the strategy, build order, and deck selection.
