import numpy as np

class ObservationEncoder:
    """
    Encodes the complex nested JSON observation dictionary from the Kaggle `cabt` engine
    into a flat numeric float32 vector suitable for a PyTorch Neural Network.
    """
    
    def __init__(self):
        # We will define the size dynamically based on features extracted.
        pass

    def encode(self, obs_dict: dict) -> np.ndarray:
        if obs_dict is None or obs_dict.get("current") is None:
            # Return dummy state for terminal/error states
            return np.zeros(self.get_state_dim(), dtype=np.float32)

        curr = obs_dict["current"]
        your_idx = curr.get("yourIndex", 0)
        opp_idx = 1 - your_idx
        
        players = curr.get("players", [])
        if len(players) < 2:
            return np.zeros(self.get_state_dim(), dtype=np.float32)
            
        p_me = players[your_idx]
        p_opp = players[opp_idx]

        # Global state
        features = [
            float(curr.get("turn", 0)),
            float(curr.get("supporterPlayed", False)),
            float(curr.get("energyAttached", False)),
            float(curr.get("retreated", False)),
        ]

        # Player state
        features.extend([
            float(p_me.get("deckCount", 0)),
            float(p_me.get("handCount", 0)),
            float(len(p_me.get("prize", []))),
            float(len(p_me.get("active", []))),
            float(len(p_me.get("bench", []))),
            float(p_me.get("poisoned", False)),
            float(p_me.get("burned", False)),
            float(p_me.get("asleep", False)),
            float(p_me.get("paralyzed", False)),
            float(p_me.get("confused", False)),
        ])

        # Opponent state
        features.extend([
            float(p_opp.get("deckCount", 0)),
            float(p_opp.get("handCount", 0)),
            float(len(p_opp.get("prize", []))),
            float(len(p_opp.get("active", []))),
            float(len(p_opp.get("bench", []))),
            float(p_opp.get("poisoned", False)),
            float(p_opp.get("burned", False)),
            float(p_opp.get("asleep", False)),
            float(p_opp.get("paralyzed", False)),
            float(p_opp.get("confused", False)),
        ])

        return np.array(features, dtype=np.float32)

    def get_state_dim(self) -> int:
        """
        Returns the flat size of the encoded state.
        Currently: 4 (Global) + 10 (Me) + 10 (Opponent) = 24
        """
        return 24
