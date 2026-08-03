def _compute_board_utility(player_dict: dict) -> float:
    u = 0.0
    active = player_dict.get("active", [])
    bench = player_dict.get("bench", [])
    
    for pkmn in active + bench:
        max_hp = pkmn.get("maxHp", 0)
        hp = pkmn.get("hp", 0)
        u += (max_hp / 100.0)
        u -= ((max_hp - hp) / 50.0)
        
        energies = pkmn.get("energyCards", [])
        if isinstance(energies, list):
            u += 0.5 * len(energies)
            
    if active:
        a = active[0]
        if a.get("hp", 100) <= 50:
            u -= 2.0
            
    return u

def calculate_reward(prev_obs: dict, curr_obs: dict, done: bool) -> float:
    """
    Shaped reward function for PTCG RL agent.
    - +1.0 for winning, -1.0 for losing
    - +0.1 for taking a prize card
    """
    reward = 0.0
    
    if curr_obs is None or prev_obs is None:
        return reward
        
    curr = curr_obs.get("current")
    prev = prev_obs.get("current")
    if not curr or not prev:
        return reward

    # Terminal state reward
    if done:
        result = curr.get("result", -1)
        if result == 0:
            reward += 1.0  # Player 0 wins
        elif result == 1:
            reward -= 1.0 # Player 1 wins
        elif result == 2:
            reward -= 0.1 # Draw (slight penalty to encourage winning)
            
        # Heavy Bench-out & Deck-out penalty
        p0_players = curr.get("players", [])
        if len(p0_players) >= 2:
            p0_deck = p0_players[0].get("deckCount", 0)
            if result == 1 and p0_deck == 0:
                reward -= 0.5 # Massive deck-out penalty
                
            if result == 1 and len(p0_players[0].get("active", [])) == 0:
                reward -= 0.5 # Extra penalty for getting benched out
            if result == 0 and len(p0_players[1].get("active", [])) == 0:
                reward += 0.5 # Extra reward for benching out opponent
            
    # Intermediate shaped rewards
    prev_players = prev.get("players", [])
    curr_players = curr.get("players", [])
    if len(prev_players) >= 2 and len(curr_players) >= 2:
        # Player 0 is RL agent, Player 1 is opponent
        p0_prev, p1_prev = prev_players[0], prev_players[1]
        p0_curr, p1_curr = curr_players[0], curr_players[1]
        
        # 1. Taking Prize Cards
        p0_prizes_taken = len(p0_prev.get("prize") or []) - len(p0_curr.get("prize") or [])
        if p0_prizes_taken > 0:
            reward += 0.2 * p0_prizes_taken
            if p0_prizes_taken >= 2:
                reward += 0.1
            
        p1_prizes_taken = len(p1_prev.get("prize") or []) - len(p1_curr.get("prize") or [])
        if p1_prizes_taken > 0:
            reward -= 0.1 * p1_prizes_taken
            if p1_prizes_taken >= 2:
                reward -= 0.1
                
        # 2. Pareto Board Utility Shaping
        p0_u_prev = _compute_board_utility(p0_prev)
        p0_u_curr = _compute_board_utility(p0_curr)
        if p0_u_curr > p0_u_prev:
            reward += 0.05 * (p0_u_curr - p0_u_prev)
        elif p0_u_curr < p0_u_prev:
            reward -= 0.05 * (p0_u_prev - p0_u_curr)
            
        # 3. Deck preservation
        p0_deck_curr = p0_curr.get("deckCount", 0)
        p0_deck_prev = p0_prev.get("deckCount", 0)
        if p0_deck_curr < p0_deck_prev and p0_deck_curr <= 10:
            reward -= 0.05 * (p0_deck_prev - p0_deck_curr)
            
    return float(reward)
