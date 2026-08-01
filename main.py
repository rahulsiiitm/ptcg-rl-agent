from src.agent.rule_based_bellibolt import rule_based_bellibolt

# Phase 4: Iono's Bellibolt ex / Kilowattrel rule-based agent.
# To swap back to the Phase 2/3 PPO policy, replace the import above with:
#   from src.agent.policy import policy_agent
# and return policy_agent(obs_dict) below.
def agent(obs_dict: dict) -> list[int]:
    return rule_based_bellibolt(obs_dict)
