from src.agent.rule_based import rule_based_agent

# For Phase 1, just use the rule-based agent as the main entrypoint.
def agent(obs_dict: dict) -> list[int]:
    return rule_based_agent(obs_dict)
