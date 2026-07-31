import sys
import os

sys.path.append(os.getcwd())

from src.env.fast_sim import FastPTCGEnv
from src.agent.policy import policy_agent

def run_test_match():
    print("Testing policy_agent against rule_based_agent using local cg engine...")
    deck = [33]*4 + [35]*4 + [47]*4 + [3]*48
    env = FastPTCGEnv(rl_deck=deck, opp_deck=deck)
    
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    
    while not done:
        # Use our trained policy agent to select an action
        action = policy_agent(obs)
        
        # Step environment
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step += 1
        
    print(f"Match Finished in {step} turns!")
    print(f"Total Shaped Reward: {total_reward:.2f}")
    
    # Evaluate who won
    result = obs.get("current", {}).get("result", -1)
    if result == 0:
        print("Result: RL Agent WON!")
    elif result == 1:
        print("Result: RL Agent LOST.")
    else:
        print("Result: DRAW.")

if __name__ == "__main__":
    run_test_match()
