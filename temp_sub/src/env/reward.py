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
            return 1.0  # Player 0 wins
        elif result == 1:
            return -1.0 # Player 1 wins
        elif result == 2:
            return -0.1 # Draw (slight penalty to encourage winning)
            
    # Intermediate shaped rewards
    prev_players = prev.get("players", [])
    curr_players = curr.get("players", [])
    if len(prev_players) >= 2 and len(curr_players) >= 2:
        # Player 0 is RL agent, Player 1 is opponent
        p0_prev, p1_prev = prev_players[0], prev_players[1]
        p0_curr, p1_curr = curr_players[0], curr_players[1]
        
        # 1. Taking Prize Cards (diff in prize array length)
        # Prizes start at 6 elements. Length goes down when prizes are taken.
        p0_prizes_taken = len(p0_prev.get("prize", [])) - len(p0_curr.get("prize", []))
        if p0_prizes_taken > 0:
            reward += 0.1 * p0_prizes_taken
            
        p1_prizes_taken = len(p1_prev.get("prize", [])) - len(p1_curr.get("prize", []))
        if p1_prizes_taken > 0:
            reward -= 0.1 * p1_prizes_taken
            
    return float(reward)
