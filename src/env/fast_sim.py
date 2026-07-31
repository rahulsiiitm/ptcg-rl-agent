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
    The opponent (Player 1) is automatically driven by the rule-based agent.
    """
    
    def __init__(self, rl_deck, opp_deck):
        super().__init__()
        self.rl_deck = rl_deck
        self.opp_deck = opp_deck
        
        # We don't define a fixed action_space or observation_space here yet
        # because the state will be encoded by the StateEncoder in the training loop
        # and actions are dynamically masked.
        self.action_space = gym.spaces.Discrete(1) # Placeholder
        self.observation_space = gym.spaces.Dict({}) # Placeholder
        
        self.last_obs = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # The game engine expects the deck as input to battle_start
        obs, sd = battle_start(self.rl_deck, self.opp_deck)
        
        if sd.errorType != 0:
            raise RuntimeError(f"Engine failed to start. Error code: {sd.errorType}")
            
        self.last_obs = obs
        self.step_count = 1
        
        # Fast-forward through opponent turns if necessary
        obs, done = self._fast_forward_opponent(obs)
        obs["step"] = self.step_count
        self.last_obs = obs
        
        return obs, {}

    def step(self, action: list[int]):
        """
        Takes the RL agent's action and advances the environment.
        """
        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before calling step()")
        
        self.step_count += 1
        prev_obs = self.last_obs
        
        # 1. Apply RL agent's action
        try:
            obs = battle_select(action)
            obs["step"] = self.step_count
        except Exception as e:
            # If the engine crashes due to an invalid action, the game terminates with a penalty.
            return self.last_obs, -1.0, True, False, {"error": str(e)}

        # 2. Fast-forward through opponent turns until it's our turn again or game over
        obs, done = self._fast_forward_opponent(obs)
        self.last_obs = obs
        
        # 3. Calculate Reward
        reward = calculate_reward(prev_obs, self.last_obs, done)
        
        # 4. Info dict
        info = {}
        
        return self.last_obs, reward, done, False, info

    def _fast_forward_opponent(self, obs):
        """
        Loops through the engine state. If it's Player 1's turn, it queries the rule-based agent.
        Returns the next observation where it's Player 0's turn, or done=True.
        """
        done = False
        while True:
            # Check if game is over
            current = obs.get("current")
            if current is None:
                # Game over
                return obs, True
                
            result = current.get("result", -1)
            if result != -1:
                # Game over (result=0 is Player 0 wins, result=1 is Player 1 wins, result=2 is Draw)
                return obs, True
                
            your_index = current.get("yourIndex", 0)
            
            if your_index == 0:
                # It's our turn, yield back to RL agent
                return obs, False
                
            # It's opponent's turn. Query rule-based agent.
            try:
                opp_action = rule_based_agent(obs)
                obs = battle_select(opp_action)
            except Exception as e:
                # If rule based agent crashes, the game is over and RL agent wins by default.
                print(f"Opponent agent crashed: {e}")
                # We can mock a result indicating Player 0 win.
                obs["current"]["result"] = 0 
                return obs, True
