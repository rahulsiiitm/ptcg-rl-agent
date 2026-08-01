"""
Phase 5 Hybrid Agent
Uses PPO Actor logits but falls back to rule_based_bellibolt if confidence is below threshold
or if it's a Card-select / YesNo sub-prompt.
"""
import os
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.rule_based_bellibolt import rule_based_bellibolt, _read_deck, SELECT_MAIN

# Hyperparameters
CONFIDENCE_THRESHOLD = 0.65

# Global singletons
_encoder = None
_model = None

def _get_model(state_dim, action_dim):
    import torch
    import torch.nn as nn
    
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
            
    return Actor(state_dim, action_dim)

def hybrid_agent(obs_dict: dict) -> list[int]:
    """
    Kaggle entrypoint.
    Turn 0: return deck.
    Sub-prompts: rule_based_bellibolt.
    Main phase: RL policy, fallback to rule_based if confidence < CONFIDENCE_THRESHOLD.
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

    # ── Sub-prompts use Rule-Based always ──
    sel_type = select_data.get('type', -1)
    if sel_type != SELECT_MAIN:
        return rule_based_bellibolt(obs_dict)

    # ── Main phase: try policy ──
    import torch
    import numpy as np
    from src.agent.state_encoder import ObservationEncoder
    from src.agent.action_mask import get_action_mask, MAX_ACTION_SPACE

    try:
        if _encoder is None:
            _encoder = ObservationEncoder()

        if _model is None:
            _model = _get_model(ObservationEncoder.STATE_DIM, MAX_ACTION_SPACE)
            
            for candidate in [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "ppo_phase3.pth"),
                "/kaggle_simulations/agent/models/ppo_phase3.pth",
            ]:
                if os.path.exists(candidate):
                    try:
                        # We load state_dict. The checkpoint comes from ActorCritic which has 'shared' and 'actor' and 'critic'.
                        _model.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True), strict=False)
                        break
                    except Exception as e:
                        print(f"Failed to load {candidate}: {e}")
                        
            _model.eval()

        # ── Encode + forward + confidence check ──
        state = _encoder.encode(obs_dict)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = _model(state_t).squeeze(0).numpy()
            
        mask = get_action_mask(obs_dict)
        
        # Apply mask
        masked_logits = logits.copy()
        masked_logits[~mask] = -1e9
        
        # Calculate softmax on masked logits
        max_logit = np.max(masked_logits)
        if max_logit == -1e9:
            # Failsafe if everything masked
            return rule_based_bellibolt(obs_dict)
            
        exp_logits = np.exp(masked_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        confidence = np.max(probs)
        best_action = int(np.argmax(probs))
        
        if confidence >= CONFIDENCE_THRESHOLD:
            return [best_action]
        else:
            return rule_based_bellibolt(obs_dict)
            
    except Exception as e:
        print(f"Hybrid agent exception: {e}")
        # Failsafe: fall back to rule-based
        return rule_based_bellibolt(obs_dict)
