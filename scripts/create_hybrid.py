import os

with open(r'd:\Projects\4th Year\ptcg-rl-agent\scripts\user_pasted_code.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

code_lines = []
for line in lines:
    if '<USER_REQUEST>' in line or '</USER_REQUEST>' in line or '<ADDITIONAL_METADATA>' in line or 'this sample code is given' in line:
        continue
    if line.startswith('device = torch.device('): # End of agent code, start of training loop
        break
    code_lines.append(line)

agent_code = ''.join(code_lines)

hybrid_wrapper = """
# ================= HYBRID ROUTER =================
from src.agent.rule_based_lucario import agent as rule_based_agent

# Global instances to avoid re-initialization
_hybrid_model = None

def get_hybrid_model(device_name="cpu"):
    global _hybrid_model
    if _hybrid_model is None:
        device = torch.device(device_name)
        _hybrid_model = MyModel(128, 2, 256, 1, 1)
        _hybrid_model.to(device)
        
        # Try to load weights if they exist
        weight_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_best.pth")
            
        if os.path.exists(weight_path):
            _hybrid_model.load_state_dict(torch.load(weight_path, map_location=device))
        _hybrid_model.eval()
    return _hybrid_model

def agent(obs_dict, configuration=None):
    try:
        obs = to_observation_class(obs_dict)
        
        # Use Rule-Based Heuristic for simple or buggy contexts (e.g., TO_ACTIVE, PRIZE, SETUP)
        if obs.select is None or obs.select.context != SelectContext.MAIN:
            return rule_based_agent(obs_dict, configuration)
            
        # Use MCTS + Transformer for the complex MAIN phase
        model = get_hybrid_model("cpu") # Kaggle uses CPU for inference
        
        # We need the deck for MCTS state sampling.
        # Load from deck.csv
        deck_path = os.path.join(os.path.dirname(__file__), "deck.csv")
        if not os.path.exists(deck_path):
            deck_path = "/kaggle_simulations/agent/deck.csv"
        if not os.path.exists(deck_path):
            deck_path = "deck.csv"
            
        my_deck = []
        if os.path.exists(deck_path):
            with open(deck_path, "r", encoding="utf-8") as f:
                my_deck = [int(line) for line in f.read().splitlines() if line.strip()]
            
        # Hack to reduce search count on Kaggle CPU so we don't timeout (10 min total per match)
        global SEARCH_COUNT
        SEARCH_COUNT = 3 # Heavily restricted for CPU.
        
        selected, _ = mcts_agent(obs_dict, my_deck, model)
        return selected
    except Exception as e:
        # Fallback to rule-based if anything crashes in the neural network
        return rule_based_agent(obs_dict, configuration)
"""

with open(r'd:\Projects\4th Year\ptcg-rl-agent\src\agent\hybrid_lucario.py', 'w', encoding='utf-8') as f:
    f.write(agent_code + hybrid_wrapper)

print('Hybrid agent generated successfully.')
