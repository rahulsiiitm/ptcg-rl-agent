"""
Phase 3 policy — PyTorch inference directly in Kaggle container.
The cabt container is FROM gcr.io/kaggle-images/python:v163 which has torch.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import sample_valid_action, MAX_ACTION_SPACE

# Global singletons
_encoder = None
_weights = None   # NumPy weights dict (loaded from .npz)


def _read_deck() -> list[int]:
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "deck.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    with open(file_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    return [int(x) for x in lines[:60]]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _numpy_forward(state: np.ndarray, w: dict) -> np.ndarray:
    """Pure-NumPy actor forward pass matching train_ppo_parallel.py ActorCritic."""
    x = relu(state @ w['s1_w'].T + w['s1_b'])
    x = relu(x    @ w['s2_w'].T + w['s2_b'])
    logits = x @ w['a_w'].T + w['a_b']
    return logits


def policy_agent(obs_dict: dict) -> list[int]:
    """
    Kaggle entrypoint. Called every turn.
    Turn 0: return deck. Every other turn: return action indices.
    """
    global _encoder, _weights

    # ── Turn 0: submit deck ──
    step = obs_dict.get("step", 1) if obs_dict else 0
    if step == 0 or obs_dict is None or obs_dict.get("current") is None:
        return _read_deck()

    select_data = obs_dict.get("select")
    if not select_data:
        return []

    options = select_data.get("option", [])
    if not options:
        return []

    # ── Lazy init ──
    if _encoder is None:
        _encoder = ObservationEncoder()

    if _weights is None:
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "ppo_weights.npz"),
            "/kaggle_simulations/agent/models/ppo_weights.npz",
        ]:
            if os.path.exists(candidate):
                _weights = dict(np.load(candidate))
                break

        if _weights is None:
            # Fallback: random weights (better than crashing)
            H, S = 256, ObservationEncoder.STATE_DIM
            _weights = {
                's1_w': np.random.randn(H, S).astype(np.float32) * 0.01,
                's1_b': np.zeros(H, dtype=np.float32),
                's2_w': np.random.randn(H, H).astype(np.float32) * 0.01,
                's2_b': np.zeros(H, dtype=np.float32),
                'a_w':  np.random.randn(MAX_ACTION_SPACE, H).astype(np.float32) * 0.01,
                'a_b':  np.zeros(MAX_ACTION_SPACE, dtype=np.float32),
            }

    # ── Encode + forward + mask sample ──
    try:
        state = _encoder.encode(obs_dict)
        logits = _numpy_forward(state, _weights)
        return sample_valid_action(logits, obs_dict)
    except Exception:
        # Failsafe: never crash — return first legal action
        return [0]
