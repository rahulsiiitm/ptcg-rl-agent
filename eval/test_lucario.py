import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.fast_sim import FastPTCGEnv
from src.agent.rule_based_lucario import agent as lucario_agent
from src.agent.rule_based_generic import rule_based_generic_agent as generic_agent

def load_deck(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]

lucario_deck = load_deck("decks/mega_lucario_ids.csv")
meta_deck_path = "decks/meta_Charizard_ex.csv"
meta_deck = load_deck(meta_deck_path)

env = FastPTCGEnv(lucario_deck, meta_deck)
env.set_opponent_agent(generic_agent)

wins, draws, losses = 0, 0, 0
matches = 50

print(f"Testing Rule-Based Lucario vs {os.path.basename(meta_deck_path)} for {matches} matches...")

for m in range(matches):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = lucario_agent(obs)
        obs, reward, done, _, info = env.step(action)
        
    result = obs.get("current", {}).get("result", -1)
    if result == 0:
        wins += 1
    elif result == 1:
        losses += 1
    else:
        draws += 1
        
    print(f"Match {m+1}/{matches} | Wins: {wins}, Losses: {losses}, Draws: {draws}")

print(f"Final Win Rate vs Meta: {wins/matches:.1%} (W:{wins} L:{losses} D:{draws})")
