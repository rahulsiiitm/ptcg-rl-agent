import numpy as np

class OpponentModel:
    """
    Parses game logs every step to track the opponent's unseen cards (deck/hand probability).
    Currently implemented as a skeleton for Phase 2 baseline.
    """
    def __init__(self):
        self.opponent_discarded_cards = 0
        self.opponent_energies_played = 0

    def update(self, obs_dict: dict):
        if not obs_dict:
            return
            
        logs = obs_dict.get("logs", [])
        for log_entry in logs:
            # logs are usually strings or dicts depending on Kaggle's engine serialization
            log_str = str(log_entry).lower()
            
            # Simple heuristic tracking (these are mock strings, adjust based on actual log format)
            if "discard" in log_str and "opponent" in log_str:
                self.opponent_discarded_cards += 1
            if "attach" in log_str and "energy" in log_str and "opponent" in log_str:
                self.opponent_energies_played += 1

    def get_features(self) -> np.ndarray:
        """
        Returns a flat numeric array representing the opponent's modeled state.
        """
        return np.array([
            float(self.opponent_discarded_cards),
            float(self.opponent_energies_played)
        ], dtype=np.float32)

    def get_feature_dim(self) -> int:
        return 2
