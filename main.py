import sys
import os

# Ensure the src directory is in the path
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.append('/kaggle_simulations/agent')

from src.agent.hybrid_lucario import agent as hybrid_agent

def agent(obs_dict):
    """
    Kaggle entrypoint for the PTCG environment (Hybrid Lucario Agent).
    """
    return hybrid_agent(obs_dict)
