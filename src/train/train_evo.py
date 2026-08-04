import os
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor

try:
    from kaggle_environments import make
except ImportError:
    print("Please install kaggle_environments: pip install kaggle_environments")
    sys.exit(1)

BASE_AGENT = "src/agent/rule_based_lucario.py"
TEMP_DIR = "temp_evo"
MUTATE_KEYS = [
    "play_pokemon_base", "play_dusk_pad", "play_switch", "play_premium",
    "play_boss", "play_carmine", "play_lillie", "attach_hero_cape",
    "evolve_base", "ability_base", "retreat_base", "attack_base"
]

def load_base_weights():
    # Provide defaults matching current heuristic
    return {
        "play_pokemon_base": 20000, "play_dusk_pad": 8000, "play_switch": 6000,
        "play_premium": 5000, "play_boss": 3200, "play_carmine": 3000,
        "play_lillie": 3100, "attach_hero_cape": 7000, "evolve_base": 9000,
        "ability_base": 30000, "retreat_base": 2000, "attack_base": 1000
    }

def mutate(base, mutation_rate=0.2, scale=300):
    variant = base.copy()
    for k in MUTATE_KEYS:
        if random.random() < mutation_rate:
            variant[k] += int(random.gauss(0, scale))
    return variant

def run_match(args):
    agent1_path, agent2_path = args
    env = make("pokemon-tcg-ai-battle")
    try:
        steps = env.run([agent1_path, agent2_path])
        res = steps[-1][0]["observation"].get("result", -1)
        return res  # 0 if agent1 won, 1 if agent2 won, 2 for draw
    except Exception:
        return -1

def main():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    import time
    start_time = time.time()
    MAX_TIME_SECONDS = 5.5 * 3600 # 5.5 hours safety cutoff
        
    base = load_base_weights()
    generations = 30
    matches_per_variant = 20
    num_variants = 6
    
    current_best = base
    
    # We will write weights to current_best.json and run it as agent1
    best_json = os.path.join(TEMP_DIR, "best_w.json")
    best_script = os.path.join(TEMP_DIR, "best_agent.py")
    
    with open(BASE_AGENT, "r") as f:
        agent_code = f.read()
        
    for gen in range(generations):
        if time.time() - start_time > MAX_TIME_SECONDS:
            print("Time limit reached. Halting evolution early to save weights.")
            break
            
        print(f"\n=== Generation {gen+1} ===")
        with open(best_json, "w") as f:
            json.dump(current_best, f)
        
        # Patch the best script to load best_json
        patched = agent_code.replace('lucario_w.json', os.path.abspath(best_json).replace('\\', '/'))
        with open(best_script, "w") as f:
            f.write(patched)
            
        variants = [mutate(current_best) for _ in range(num_variants)]
        
        variant_scores = [0] * num_variants
        tasks = []
        for v_idx, variant in enumerate(variants):
            v_json = os.path.join(TEMP_DIR, f"v{v_idx}_w.json")
            v_script = os.path.join(TEMP_DIR, f"v{v_idx}_agent.py")
            with open(v_json, "w") as f:
                json.dump(variant, f)
            with open(v_script, "w") as f:
                f.write(agent_code.replace('lucario_w.json', os.path.abspath(v_json).replace('\\', '/')))
                
            for _ in range(matches_per_variant):
                # We play 50% as player 1 and 50% as player 2
                if random.random() < 0.5:
                    tasks.append( (best_script, v_script, v_idx, "p2") )
                else:
                    tasks.append( (v_script, best_script, v_idx, "p1") )
                    
        print(f"Running {len(tasks)} matches...")
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_match, [(t[0], t[1]) for t in tasks]))
            
        for (a1, a2, v_idx, role), result in zip(tasks, results):
            if result == -1 or result == 2: continue # Crash or Draw
            if role == "p2" and result == 1:
                variant_scores[v_idx] += 1
            elif role == "p1" and result == 0:
                variant_scores[v_idx] += 1
                
        for i, score in enumerate(variant_scores):
            win_rate = score / matches_per_variant
            print(f"Variant {i} winrate: {win_rate:.2f}")
            
        best_v = max(range(num_variants), key=lambda i: variant_scores[i])
        if variant_scores[best_v] > matches_per_variant / 2:
            print(f"Variant {best_v} outperformed base! Adopting new weights.")
            current_best = variants[best_v]
        else:
            print("No variant beat the base model. Retaining base weights.")
            
    print("\nEvolution complete! Best weights:")
    print(json.dumps(current_best, indent=2))
    
    with open("lucario_w.json", "w") as f:
        json.dump(current_best, f, indent=2)
    print("Saved to lucario_w.json")

if __name__ == "__main__":
    main()
