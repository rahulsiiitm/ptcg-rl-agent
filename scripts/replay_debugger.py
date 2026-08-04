import os
import sys
import json
import argparse

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rule_based_lucario import agent as lucario_agent
try:
    from cg.api import all_card_data
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
except ImportError:
    card_table = {}

def format_card(card_id):
    if not card_table:
        return str(card_id)
    c = card_table.get(card_id)
    if not c:
        return str(card_id)
    return f"{c.name} (ID: {card_id})"

def analyze_replay(file_path):
    print(f"--- Replay Debugger: {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data.get("steps", data) if isinstance(data, dict) else data
    
    for step_idx, step_data in enumerate(steps):
        if not step_data or not isinstance(step_data, list):
            continue
            
        # Kaggle JSON often has 2 items per step, one for each player
        for player_idx in [0, 1]:
            if len(step_data) <= player_idx:
                continue
                
            player_step = step_data[player_idx]
            obs = player_step.get("observation", {})
            action = player_step.get("action")
            
            current = obs.get("current", {})
            if not current:
                continue
                
            your_idx = current.get("yourIndex", -1)
            
            # Print state if this is the active turn
            if action is not None and your_idx != -1:
                turn = current.get("turn", 0)
                
                # Fetch what our rule-based agent WOULD have done
                try:
                    simulated_action = lucario_agent(obs)
                except Exception as e:
                    simulated_action = f"ERROR: {e}"
                
                players = current.get("players", [])
                if len(players) > your_idx:
                    p = players[your_idx]
                    active = p.get("active", [])
                    active_name = "None"
                    if active and active[0]:
                        active_name = format_card(active[0].get("id"))
                        
                    print(f"\n[Step {step_idx} | Turn {turn}] Player {player_idx}")
                    print(f"  Active: {active_name}")
                    print(f"  Played Action:    {action}")
                    print(f"  Lucario Agent ID: {simulated_action}")
                    
                    if simulated_action != action:
                        print(f"  => DIVERGENCE: The current Lucario agent chose a different action than the replay!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a Kaggle PTCG replay JSON.")
    parser.add_argument("replay_file", type=str, help="Path to replay.json")
    args = parser.parse_args()
    
    if os.path.exists(args.replay_file):
        analyze_replay(args.replay_file)
    else:
        print(f"File not found: {args.replay_file}")
