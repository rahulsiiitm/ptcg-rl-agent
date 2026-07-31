import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym

# Add parent dir to sys path for Kaggle imports
import sys
sys.path.append(os.getcwd())

from src.env.fast_sim import FastPTCGEnv
from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import get_action_mask, MAX_ACTION_SPACE

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        return self.actor(state), self.critic(state)

def train_ppo(episodes=50, max_steps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Environment & Decks
    # For baseline training, we use the same mono-water deck we tested.
    deck = [33]*4 + [35]*4 + [47]*4 + [3]*48
    env = FastPTCGEnv(rl_deck=deck, opp_deck=deck)
    
    # 2. Components
    encoder = ObservationEncoder()
    state_dim = encoder.get_state_dim()
    
    model = ActorCritic(state_dim, MAX_ACTION_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    gamma = 0.99
    clip_ratio = 0.2
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0
        
        while not done and step < max_steps:
            state_vec = encoder.encode(obs)
            state_t = torch.tensor(state_vec, dtype=torch.float32).to(device).unsqueeze(0)
            
            # Action mask
            mask = get_action_mask(obs).to(device)
            
            # Forward pass
            logits, value = model(state_t)
            
            # Apply mask
            masked_logits = logits.clone()
            masked_logits[0, ~mask] = -1e9
            
            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # We sample a single integer action as a simplification.
            # In a real variable-length scenario, we'd sample maxCount items.
            # The Kaggle env `battle_select` requires a list.
            try:
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                action = [action_idx.item()]
            except ValueError:
                # Fallback if masking completely fails (e.g. no valid options)
                action = []
            
            next_obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            
            # --- PPO Update (Simplified for single step, usually done in batches) ---
            # Next value
            if not done:
                next_state_vec = encoder.encode(next_obs)
                next_state_t = torch.tensor(next_state_vec, dtype=torch.float32).to(device).unsqueeze(0)
                _, next_value = model(next_state_t)
                td_target = reward + gamma * next_value.detach()
            else:
                td_target = torch.tensor([[reward]], dtype=torch.float32).to(device)
                
            advantage = td_target - value
            
            # Critic loss
            critic_loss = advantage.pow(2).mean()
            
            # Actor loss
            # (In a real PPO, we'd compute ratio with old_log_probs from rollouts)
            actor_loss = -(log_prob * advantage.detach()).mean()
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            obs = next_obs
            step += 1
            
        print(f"Episode {ep+1}/{episodes} - Reward: {ep_reward:.3f} - Steps: {step}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ppo_baseline.pth")
    print("Saved baseline PPO model.")

if __name__ == "__main__":
    train_ppo()
