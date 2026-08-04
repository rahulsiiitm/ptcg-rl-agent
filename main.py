import sys
import os

# Ensure the src directory is in the path
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.append('/kaggle_simulations/agent')

from src.agent.rule_based_lucario import agent as lucario_agent

def agent(obs_dict):
    """
    Kaggle entrypoint for the PTCG environment (Rule-Based Lucario Agent).
    """
    return lucario_agent(obs_dict)
