import sys
import os

# Ensure the src directory is in the path
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.append('/kaggle_simulations/agent')

from src.agent.policy import policy_agent

def agent(obs_dict):
    """
    Kaggle entrypoint for the PTCG environment (RL Agent).
    """
    return policy_agent(obs_dict)
