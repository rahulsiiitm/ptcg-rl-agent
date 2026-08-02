import numpy as np

MAX_ACTION_SPACE = 150

def get_action_mask(obs_dict: dict) -> np.ndarray:
    """
    Given the observation dictionary, returns a boolean mask of shape (MAX_ACTION_SPACE,)
    where True means the action index is valid.
    """
    mask = np.zeros(MAX_ACTION_SPACE, dtype=bool)
    
    if obs_dict is None:
        return mask
        
    select_data = obs_dict.get("select")
    if not select_data:
        return mask
        
    options = select_data.get("option", [])
    num_options = min(len(options), MAX_ACTION_SPACE)
    
    if num_options > 0:
        mask[:num_options] = True
        
        # Heuristic Masking: Prevent skipping attacks.
        # If the agent can ATTACK (type 13), mask out PASS_TURN (type 14).
        has_attack = False
        pass_turn_idx = -1
        
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                opt_type = opt.get("type")
                if opt_type == 13: # ATTACK
                    has_attack = True
                elif opt_type == 14: # END TURN / PASS
                    pass_turn_idx = i
                    
        if has_attack and pass_turn_idx != -1:
            mask[pass_turn_idx] = False
            
    return mask

def sample_valid_action(logits: np.ndarray, obs_dict: dict) -> list[int]:
    """
    Takes raw logits (from policy network) and the observation dict.
    Masks out invalid actions, applies softmax, and samples `maxCount` items
    without replacement.
    """
    select_data = obs_dict.get("select")
    if not select_data:
        return []
        
    options = select_data.get("option", [])
    if not options:
        return []
        
    max_count = select_data.get("maxCount", 1)
    max_count = min(max_count, len(options))
    
    mask = get_action_mask(obs_dict)
    
    masked_logits = np.copy(logits)
    masked_logits[~mask] = -1e9
    
    # Softmax
    e_x = np.exp(masked_logits - np.max(masked_logits))
    probs = e_x / e_x.sum()
    
    # Sample without replacement
    action_indices = np.random.choice(len(probs), size=max_count, replace=False, p=probs)
    return action_indices.tolist()
