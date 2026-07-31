import os
import sys
import numpy as np

# Add parent path so imports work in Kaggle
sys.path.append(os.getcwd())

from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import sample_valid_action, MAX_ACTION_SPACE

# Global instances (instantiated once per match)
_weights = None
_encoder = None

def _read_deck() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def relu(x):
    return np.maximum(0, x)

def numpy_actor(state, weights):
    # L1
    x = np.dot(state, weights['l1_w'].T) + weights['l1_b']
    x = relu(x)
    # L2
    x = np.dot(x, weights['l2_w'].T) + weights['l2_b']
    x = relu(x)
    # L3
    x = np.dot(x, weights['l3_w'].T) + weights['l3_b']
    return x

def policy_agent(obs_dict: dict) -> list[int]:
    """
    Kaggle entrypoint for the RL agent. Pure Numpy!
    """
    global _weights, _encoder
    
    # 1. Check if we need to return deck
    step = obs_dict.get("step", 0)
    if step == 0:
        return _read_deck()
        
    # 2. Check if it's not our turn
    select_data = obs_dict.get("select")
    if not select_data:
        return []
        
    options = select_data.get("option", [])
    if not options:
        return []

    # 3. Initialize model if not already done
    if _weights is None:
        _encoder = ObservationEncoder()
        model_path = os.path.join(os.getcwd(), "models", "ppo_weights.npz")
        if not os.path.exists(model_path):
            model_path = "/kaggle_simulations/agent/models/ppo_weights.npz"
            
        if os.path.exists(model_path):
            _weights = np.load(model_path)
        else:
            # Fallback random initialization if weights are missing
            _weights = {
                'l1_w': np.random.randn(128, 24).astype(np.float32) * 0.1,
                'l1_b': np.zeros(128, dtype=np.float32),
                'l2_w': np.random.randn(128, 128).astype(np.float32) * 0.1,
                'l2_b': np.zeros(128, dtype=np.float32),
                'l3_w': np.random.randn(14, 128).astype(np.float32) * 0.1,
                'l3_b': np.zeros(14, dtype=np.float32),
            }

    # 4. Encode state
    state_vec = _encoder.encode(obs_dict)
    state_arr = np.array(state_vec, dtype=np.float32)
    
    # 5. Forward pass
    logits = numpy_actor(state_arr, _weights)
    
    # 6. Sample action using mask
    action = sample_valid_action(logits, obs_dict)
    return action
