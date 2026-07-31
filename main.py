from src.agent.policy import policy_agent

# For Phase 1, just use the rule-based agent as the main entrypoint.
def agent(obs_dict: dict) -> list[int]:
    return policy_agent(obs_dict)
