import random

def rule_based_agent(obs_dict: dict) -> list[int]:
    """
    A simple heuristic rule-based agent.
    obs_dict matches the Observation structure.
    obs_dict["select"] contains the selection options.
    """
    select_data = obs_dict.get("select")
    if not select_data:
        return []

    options = select_data.get("option", [])
    max_count = select_data.get("maxCount", 1)
    
    # Selection type mapping based on cabt engine API:
    # 0 = MAIN (play, attach, evolve, etc.)
    # Select types generally map to the context.
    select_type = select_data.get("type", 0)

    # Simple heuristic:
    # If we are in MAIN phase (select_type == 0):
    # Option types: 13=ATTACK, 9=EVOLVE, 7=PLAY, 8=ATTACH, 10=ABILITY, 14=END
    if select_type == 0:
        # Sort options by priority
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
            # Add some randomness to tie-break
            score += random.random()
            scored_options.append((score, i))
            
        scored_options.sort(reverse=True)
        # Select the top max_count options
        selected_indices = [i for _, i in scored_options[:max_count]]
        return selected_indices
    
    # For other selection types, just randomly sample
    if not options:
        return []
        
    num_to_select = min(max_count, len(options))
    return random.sample(list(range(len(options))), num_to_select)
