import os
import json
import glob
import numpy as np
import torch
import sys

# Ensure imports work when run from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state_encoder import ObservationEncoder
from agent.action_mask import get_action_mask, MAX_ACTION_SPACE

def parse_replays(replay_dir="data/replays"):
    encoder = ObservationEncoder()
    states = []
    masks = []
    actions = []
    
    files = glob.glob(os.path.join(replay_dir, "*.json"))
    if not files:
        # Also check root for user uploads
        files = glob.glob("*.json")
        
    print(f"Found {len(files)} replay files.")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        if isinstance(data, list):
            steps = data
        else:
            steps = data.get("steps", [])
            
        for step_idx, step_data in enumerate(steps):
            if not step_data or not isinstance(step_data, list):
                continue
                
            for player_idx in [0, 1]:
                if len(step_data) <= player_idx:
                    continue
                    
                player_step = step_data[player_idx]
                obs_dict = player_step.get("observation", {})
                action_list = player_step.get("action", [])
                
                sel = obs_dict.get("select", {})
                
                # Only imitate MAIN phase decisions where an action was actually taken
                if sel and sel.get("type") == 0 and len(action_list) > 0:
                    action_idx = action_list[0]
                    
                    if action_idx >= MAX_ACTION_SPACE:
                        print(f"Skipping action_idx {action_idx} (>= {MAX_ACTION_SPACE})")
                        continue # Ignore actions outside our fixed action space
                        
                    # Build state from the perspective of this player
                    state_vec = encoder.encode(obs_dict)
                    
                    # Build mask
                    mask = get_action_mask(obs_dict)
                    
                    # Only train on valid moves (the expert should only make valid moves)
                    if action_idx < len(mask) and mask[action_idx] == 1:
                        states.append(state_vec)
                        masks.append(mask)
                        actions.append(action_idx)
    
    if len(states) == 0:
        print("No valid MAIN phase actions found in replays.")
        return
        
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    masks_t = torch.tensor(np.array(masks), dtype=torch.bool)
    actions_t = torch.tensor(np.array(actions), dtype=torch.long)
    
    out_file = "data/replays_dataset.pt"
    os.makedirs("data", exist_ok=True)
    torch.save({
        "states": states_t,
        "masks": masks_t,
        "actions": actions_t
    }, out_file)
    print(f"Saved {len(states)} transitions to {out_file}")

if __name__ == "__main__":
    parse_replays()
