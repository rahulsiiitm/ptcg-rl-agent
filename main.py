import sys
import os

# Ensure the src directory is in the path
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.append('/kaggle_simulations/agent')

# Phase 12: Rule-based Lucario agent — proven 600+ Elo on the real Kaggle ladder.
# The hybrid MCTS approach scored only ~400 Elo on the real ladder because:
# 1. Kaggle CPU is too slow for meaningful MCTS (3 sims vs 20 during local training)
# 2. Neural network policy quality at 3 sims is worse than the hand-tuned heuristic
# Rule: NEVER replace this with a new agent unless it beats 600 Elo on the real ladder.
from src.agent.rule_based_lucario import agent as rule_based_agent

def agent(obs_dict):
    """
    Kaggle entrypoint for the PTCG environment (Rule-Based Lucario Agent).
    Consistently achieves 600+ Elo. Hybrid MCTS is trained locally but not deployed.
    """
    return rule_based_agent(obs_dict)
