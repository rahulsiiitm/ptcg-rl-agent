import gymnasium as gym
import numpy as np

# Add parent dir to path if needed so we can import from competition_files
import sys
import os
sys.path.append(os.getcwd())

from competition_files.sample_submission.sample_submission.cg.game import battle_start, battle_select
from src.agent.rule_based import rule_based_agent
from src.env.reward import calculate_reward

class FastPTCGEnv(gym.Env):
    """
    Gymnasium-compatible environment wrapping the fast C++ `cg` engine.
    The RL agent plays as Player 0.
    The opponent (Player 1) is driven by self.opponent_agent (defaults to rule_based_agent).
    """
    
    def __init__(self, rl_deck, opp_deck, opponent_agent=None):
        super().__init__()
        self.rl_deck = rl_deck
        self.opp_deck = opp_deck
        self.opponent_agent = opponent_agent if opponent_agent is not None else rule_based_agent
        
        self.action_space = gym.spaces.Discrete(1) # Placeholder
        self.observation_space = gym.spaces.Dict({}) # Placeholder
        self.last_obs = None
        
    def set_opponent_agent(self, agent_fn):
        """Allows swapping the opponent agent dynamically (e.g. for self-play pool)."""
        self.opponent_agent = agent_fn
        
    def set_opponent_deck(self, opp_deck):
        """Allows swapping the opponent deck dynamically (e.g. to train against multiple meta decks)."""
        self.opp_deck = opp_deck
        
    def set_rl_deck(self, rl_deck):
        """Allows swapping the RL agent's deck dynamically to learn generalized strategies."""
        self.rl_deck = rl_deck
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, sd = battle_start(self.rl_deck, self.opp_deck)
        if sd.errorType != 0:
            raise RuntimeError(f"Engine failed to start. Error code: {sd.errorType}")
            
        self.last_obs = obs
        self.step_count = 1
        
        obs, done = self._fast_forward_opponent(obs)
        obs["step"] = self.step_count
        self.last_obs = obs
        return obs, {}

    def step(self, action: list[int]):
        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before calling step()")
        
        self.step_count += 1
        prev_obs = self.last_obs
        
        try:
            obs = battle_select(action)
            obs["step"] = self.step_count
        except Exception as e:
            return self.last_obs, -1.0, True, False, {"error": str(e)}

        obs, done = self._fast_forward_opponent(obs)
        self.last_obs = obs
        reward = calculate_reward(prev_obs, self.last_obs, done)
        info = {}
        return self.last_obs, reward, done, False, info

    def _fast_forward_opponent(self, obs):
        done = False
        while True:
            current = obs.get("current")
            if current is None:
                return obs, True
                
            result = current.get("result", -1)
            if result != -1:
                return obs, True
                
            your_index = current.get("yourIndex", 0)
            if your_index == 0:
                return obs, False
                
            try:
                opp_action = self.opponent_agent(obs)
                obs = battle_select(opp_action)
            except Exception as e:
                print(f"Opponent agent crashed: {e}")
                obs["current"]["result"] = 0 
                return obs, True
