import os
import glob
import random
import torch
import numpy as np

from src.agent.policy import policy_agent
from src.agent.rule_based import rule_based_agent
from src.agent.state_encoder import ObservationEncoder

class SelfPlayPool:
    def __init__(self, pool_dir="models/pool", latest_model_path="models/ppo_phase3.pth"):
        self.pool_dir = pool_dir
        self.latest_model_path = latest_model_path
        
        # Probabilities for sampling
        self.prob_rule_based = 0.20
        self.prob_latest = 0.50
        # The remaining 30% is a random historical checkpoint
        
    def _get_historical_checkpoints(self):
        if not os.path.exists(self.pool_dir):
            return []
        return glob.glob(os.path.join(self.pool_dir, "*.pth"))

    def sample_opponent(self):
        """
        Samples an opponent agent function.
        Returns a callable: agent(obs) -> action_index
        """
        rand_val = random.random()
        
        # 1) Rule-based agent (20%)
        if rand_val < self.prob_rule_based:
            return rule_based_agent
            
        # 2) Latest model (50%)
        if rand_val < self.prob_rule_based + self.prob_latest:
            if os.path.exists(self.latest_model_path):
                return self._load_policy_agent(self.latest_model_path)
            else:
                return rule_based_agent # Fallback if no latest model exists yet
                
        # 3) Historical checkpoint (30%)
        checkpoints = self._get_historical_checkpoints()
        if checkpoints:
            chosen_ckpt = random.choice(checkpoints)
            return self._load_policy_agent(chosen_ckpt)
        else:
            # Fallback if no historical checkpoints exist
            if os.path.exists(self.latest_model_path):
                return self._load_policy_agent(self.latest_model_path)
            return rule_based_agent

    def _load_policy_agent(self, model_path):
        """
        Loads a PyTorch model from disk and returns an agent function that uses it.
        We instantiate a local ActorCritic network for the worker process.
        """
        # We need to import ActorCritic locally to avoid circular imports 
        # or multiprocessing pickling issues if passed from main
        from src.train.train_ppo import ActorCritic
        
        encoder = ObservationEncoder()
        state_dim = encoder.get_state_dim()
        # Action space max is 100 for now, should match MAX_ACTION_SPACE in train_ppo
        from src.agent.action_mask import MAX_ACTION_SPACE
        
        model = ActorCritic(state_dim, MAX_ACTION_SPACE)
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            return rule_based_agent
            
        model.eval()
        
        def agent(obs):
            state = encoder.encode(obs)
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            from src.agent.action_mask import get_action_mask
            mask = get_action_mask(obs)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = model(state_t)
                masked = logits.clone()
                masked[mask_t == 0] = -1e9
                probs = torch.softmax(masked, dim=-1)
                
                # If everything is masked out (should not happen if engine works)
                if probs.sum() == 0 or torch.isnan(probs).any():
                    return rule_based_agent(obs)
                    
                action = torch.multinomial(probs[0], 1).item()
                return action
                
        return agent

