import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.fast_sim import FastPTCGEnv
from src.agent.rule_based_lucario import agent as lucario_agent
from src.train.train_ppo import ActorCritic
from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import get_action_mask

def load_deck(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]

rl_deck = load_deck("decks/lopunny_froslass_ids.csv")
lucario_deck = load_deck("decks/mega_lucario_ids.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic().to(device)
model.load_state_dict(torch.load("models/pool/checkpoint_55000.pth", map_location=device, weights_only=True))
model.eval()

encoder = ObservationEncoder()

def ppo_agent(obs_dict):
    try:
        step = obs_dict.get("step", 1) if obs_dict else 0
        if step == 0:
            return rl_deck

        mask = get_action_mask(obs_dict)
        if mask.sum() == 0:
            return []
            
        state = encoder.encode(obs_dict)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _, _ = model(state_t)
            
        logits = logits.squeeze(0)
        mask_t = torch.from_numpy(mask).to(device)
        logits[~mask_t] = -1e9
        probs = torch.softmax(logits, dim=-1)
        
        select_data = obs_dict.get("select", {})
        max_count = select_data.get("maxCount", 1)
        num_valid = mask.sum()
        max_count = min(max_count, num_valid)
        
        if max_count <= 1:
            idx = torch.argmax(probs).item()
            return [idx]
        else:
            vals, inds = torch.topk(probs, max_count)
            return inds.tolist()
            
    except Exception as e:
        print("PPO error:", e)
        return [0]

env = FastPTCGEnv(rl_deck, lucario_deck)
env.set_opponent_agent(lucario_agent)

wins, draws, losses = 0, 0, 0
matches = 50

print(f"Testing 55000 checkpoint vs Lucario for {matches} matches...")

for m in range(matches):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = ppo_agent(obs)
        obs, reward, done, _, info = env.step(action)
        
    result = obs.get("current", {}).get("result", -1)
    if result == 0:
        wins += 1
    elif result == 1:
        losses += 1
    else:
        draws += 1
        
    print(f"Match {m+1}/{matches} | Wins: {wins}, Losses: {losses}, Draws: {draws}")

print(f"Final Win Rate vs Lucario: {wins/matches:.1%} (W:{wins} L:{losses} D:{draws})")
