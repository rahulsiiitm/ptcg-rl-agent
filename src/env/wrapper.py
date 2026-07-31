import gymnasium as gym
import numpy as np

class CabtEnvWrapper(gym.Env):
    """
    Gymnasium-style wrapper for the cabt engine.
    This wrapper handles the variable-sized action space by expecting 
    the agent to return a ranked list of scores for each legal option,
    or just handling it directly as a dict.
    
    Since the action space changes size every turn, a standard discrete 
    action space doesn't fit well. Instead, we define the action space 
    as a maximum length vector or simply operate in a custom loop.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        # cabt engine doesn't have a fixed observation space format suitable 
        # for standard gym spaces yet, so we return the raw obs_dict
        self.observation_space = gym.spaces.Dict({}) 
        self.action_space = gym.spaces.Discrete(1) # Placeholder

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        # Action is expected to be a list of indices chosen by the agent
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def render(self):
        return self.env.render()
