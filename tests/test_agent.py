import sys
import os
import glob
import numpy as np

sys.path.append(os.getcwd())

from src.env.fast_sim import FastPTCGEnv
from src.agent.policy import policy_agent
from src.agent.rule_based_generic import rule_based_generic_agent

def _load_deck_from_csv(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]

def run_multi_deck_test():
    rl_deck = _load_deck_from_csv("decks/lopunny_froslass_ids.csv")
    meta_deck_files = glob.glob("decks/meta_*.csv")
    
    # Pick a subset of 10 interesting meta decks to test against
    test_files = np.random.choice(meta_deck_files, min(10, len(meta_deck_files)), replace=False)
    
    print(f"Testing RL Agent (Lopunny/Snorlax) against {len(test_files)} Meta Decks (Heuristic Opponent)...")
    print("-" * 60)
    
    total_wins = 0
    total_matches = len(test_files)
    
    for df in test_files:
        opp_deck = _load_deck_from_csv(df)
        deck_name = os.path.basename(df).replace("meta_", "").replace(".csv", "")
        
        env = FastPTCGEnv(rl_deck=rl_deck, opp_deck=opp_deck)
        env.set_opponent_agent(rule_based_generic_agent)
        
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            action = policy_agent(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            
        result = obs.get("current", {}).get("result", -1)
        if result == 0:
            res_str = "WON"
            total_wins += 1
        elif result == 1:
            res_str = "LOST"
        else:
            res_str = "DRAW"
            
        print(f"vs {deck_name:<25} | Result: {res_str} | Turns: {step:<3} | Reward: {total_reward:.2f}")

    print("-" * 60)
    print(f"Overall Win Rate: {total_wins}/{total_matches} ({(total_wins/total_matches)*100:.1f}%)")

if __name__ == "__main__":
    run_multi_deck_test()
