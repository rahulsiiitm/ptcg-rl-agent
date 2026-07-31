import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from src.agent.state_encoder import ObservationEncoder
from src.env.fast_sim import FastPTCGEnv
from src.train.train_ppo_parallel import SNORLAX_DECK
import random

def test_state_encoder_shape():
    encoder = ObservationEncoder()
    assert encoder.STATE_DIM == 645
    
    # Test on empty/null dict
    state_empty = encoder.encode({})
    assert state_empty.shape == (645,), f"Expected (645,) got {state_empty.shape}"
    assert state_empty.dtype == np.float32

    # Test on real environment
    env = FastPTCGEnv(rl_deck=SNORLAX_DECK, opp_deck=SNORLAX_DECK)
    obs, info = env.reset()
    
    state = encoder.encode(obs)
    assert state.shape == (645,), f"Expected (645,) got {state.shape}"
    assert not np.isnan(state).any(), "NaN found in state!"
    
    # Random steps to populate board and verify
    for _ in range(50):
        legal = obs.get('select', {}).get('option', [])
        if not legal: break
        action = random.choice(legal)
        obs, _, done, _, _ = env.step(action)
        state_step = encoder.encode(obs)
        assert state_step.shape == (645,)
        assert not np.isnan(state_step).any()
    
    print("State Encoder Test Passed! Output Shape:", state.shape)

if __name__ == "__main__":
    test_state_encoder_shape()
