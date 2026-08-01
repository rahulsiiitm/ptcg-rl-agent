"""
Phase 3 policy — PyTorch inference directly in Kaggle container.
The cabt container is FROM gcr.io/kaggle-images/python:v163 which has torch.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import sample_valid_action, MAX_ACTION_SPACE

# Global singletons
_encoder = None
_model = None

def _read_deck() -> list[int]:
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "deck.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    with open(file_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    return [int(x) for x in lines[:60]]

class Actor(nn.Module):
    """Lightweight PyTorch Actor for inference."""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h)

def policy_agent(obs_dict: dict) -> list[int]:
    """
    Kaggle entrypoint. Called every turn.
    Turn 0: return deck. Every other turn: return action indices.
    """
    global _encoder, _model

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

    if _model is None:
        _model = Actor(ObservationEncoder.STATE_DIM, MAX_ACTION_SPACE)
        loaded = False
        
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "ppo_phase3.pth"),
            "/kaggle_simulations/agent/models/ppo_phase3.pth",
        ]:
            if os.path.exists(candidate):
                try:
                    # We load state_dict. The checkpoint comes from ActorCritic which has 'shared' and 'actor' and 'critic'.
                    # PyTorch load_state_dict with strict=False will load 'shared' and 'actor' weights perfectly!
                    _model.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True), strict=False)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {candidate}: {e}")
                    
        _model.eval()

    # ── Encode + forward + mask sample ──
    try:
        state = _encoder.encode(obs_dict)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = _model(state_t).squeeze(0).numpy()
        return sample_valid_action(logits, obs_dict)
    except Exception:
        # Failsafe: never crash — return first legal action
        return [0]
