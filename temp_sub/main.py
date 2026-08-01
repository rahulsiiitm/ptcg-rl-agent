import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.hybrid_agent import hybrid_agent

def agent(obs_dict):
    """
    Kaggle entrypoint for the PTCG environment.
    """
    return hybrid_agent(obs_dict)
