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
_models = []

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
    global _encoder, _models

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

    if not _models:
        import glob
        
        # Paths to search for models
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pool_dir = os.path.join(base_dir, "models", "pool")
        kg_pool_dir = "/kaggle_simulations/agent/models/pool"
        
        candidates = []
        
        # 1. Look for latest weights (main current weight)
        for p in [os.path.join(base_dir, "models", "ppo_latest.pth"), "/kaggle_simulations/agent/models/ppo_latest.pth"]:
            if os.path.exists(p): candidates.append(p)
            
        # 2. Grab top 3 checkpoints from pool (highest episode numbers)
        checkpoints = []
        for d in [pool_dir, kg_pool_dir]:
            if os.path.exists(d):
                checkpoints.extend(glob.glob(os.path.join(d, "checkpoint_*.pth")))
        
        # Sort by episode number (e.g. checkpoint_15000.pth -> 15000)
        def _get_ep(path):
            try:
                return int(os.path.basename(path).split('_')[1].split('.')[0])
            except:
                return -1
        checkpoints.sort(key=_get_ep, reverse=True)
        candidates.extend(checkpoints[:3])
        
        # Deduplicate
        candidates = list(dict.fromkeys(candidates))
        
        # Load up to 3 models for ensemble
        for candidate in candidates[:3]:
            try:
                m = Actor(ObservationEncoder.STATE_DIM, MAX_ACTION_SPACE)
                m.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True), strict=False)
                m.eval()
                _models.append(m)
                print(f"Loaded ensemble member: {candidate}")
            except Exception as e:
                print(f"Failed to load {candidate}: {e}")
                
        if not _models:
            # Absolute fallback
            m = Actor(ObservationEncoder.STATE_DIM, MAX_ACTION_SPACE)
            m.eval()
            _models.append(m)

    # ── Encode + forward + mask sample ──
    try:
        state = _encoder.encode(obs_dict)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        ensemble_logits = []
        with torch.no_grad():
            for m in _models:
                ensemble_logits.append(m(state_t).squeeze(0).numpy())
                
        # Average logits for ensemble
        avg_logits = np.mean(ensemble_logits, axis=0)
        return sample_valid_action(avg_logits, obs_dict)
    except Exception:
        # Failsafe: never crash — return first legal action
        return [0]
