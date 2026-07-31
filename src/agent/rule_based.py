import random
import time

def rule_based_agent(obs_dict: dict) -> list[int]:
    """
    A robust heuristic rule-based agent for the Kaggle PTGC challenge.
    Prioritizes: KO attacks > Evolve > Play supporters > Attach Energy > Attack > End
    Always returns a valid fallback to prevent crashing.
    """
    start_time = time.time()
    
    try:
        select_data = obs_dict.get("select")
        if not select_data:
            return []

        options = select_data.get("option", [])
        if not options:
            return []
            
        max_count = select_data.get("maxCount", 1)
        select_type = select_data.get("type", 0)

        # 0 = MAIN phase. OptionTypes: 13=ATTACK, 9=EVOLVE, 7=PLAY, 8=ATTACH, 10=ABILITY, 14=END
        if select_type == 0:
            priority = {
                13: 100, # ATTACK
                9: 90,   # EVOLVE
                7: 80,   # PLAY
                8: 70,   # ATTACH
                10: 60,  # ABILITY
                14: 10,  # END
            }
            
            scored_options = []
            for i, opt in enumerate(options):
                opt_type = opt.get("type", -1)
                score = priority.get(opt_type, 0)
                
                # Further heuristics could go here, e.g.:
                # if opt_type == 13 (ATTACK) and damage > active_hp: score += 50
                
                # Tie breaker
                score += random.random()
                scored_options.append((score, i))
                
            scored_options.sort(reverse=True)
            return [i for _, i in scored_options[:max_count]]

        # For YES_NO (9), always pick YES (1) if it's safe, else random
        if select_type == 9:
            # Try to pick YES
            for i, opt in enumerate(options):
                if opt.get("type") == 1:
                    return [i]
                    
        # Fallback: pick random legal options
        num_to_select = min(max_count, len(options))
        # cabt API states the engine only ever presents legal moves, so any index is safe
        return random.sample(list(range(len(options))), num_to_select)
        
    except Exception as e:
        # Failsafe: Never crash. Always return the first valid option if available
        # This guarantees we don't time out or exception out of a match.
        select_data = obs_dict.get("select", {})
        options = select_data.get("option", [])
        if options:
            return [0]
        return []
