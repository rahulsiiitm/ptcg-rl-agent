import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from data.card_lookup import CardLookup
except ImportError:
    CardLookup = None

class BayesianTracker:
    """
    Advanced Opponent Model (Phase 7).
    Parses the public board state (especially the discard pile) to track 
    how many critical "power cards" the opponent has burned. 
    """
    def __init__(self):
        self.db = CardLookup() if CardLookup is not None else None
        
        # Track counts of these key cards in the discard pile
        self.tracked_names = [
            "boss's orders", "iono", "professor's research",
            "super rod", "ultra ball", "nest ball",
            "rare candy", "switch cart", "escape rope", "mirage gate"
        ]
        
        self.discard_counts = np.zeros(len(self.tracked_names), dtype=np.float32)
        
        # Generic stats
        self.total_discarded = 0.0
        self.hand_size = 0.0
        self.deck_size = 0.0

    def update(self, obs_dict: dict):
        if not obs_dict:
            return
            
        p2 = obs_dict.get("p2", {})
        
        # Parse hand/deck sizes (if available)
        self.hand_size = float(len(p2.get("hand", [])))
        self.deck_size = float(len(p2.get("deck", [])))
        
        # Parse discard pile
        discard = p2.get("discard", [])
        self.total_discarded = float(len(discard))
        
        self.discard_counts.fill(0)
        
        if self.db:
            for card in discard:
                card_id = card.get("id") if isinstance(card, dict) else card
                if card_id:
                    c_rows = self.db.get_card(card_id)
                    if c_rows:
                        name = c_rows[0].name.lower()
                        for i, tracked in enumerate(self.tracked_names):
                            if tracked in name:
                                self.discard_counts[i] += 1.0

    def get_features(self) -> np.ndarray:
        generic = np.array([
            self.total_discarded / 60.0,
            self.hand_size / 20.0,
            self.deck_size / 60.0
        ], dtype=np.float32)
        normalized_counts = self.discard_counts / 4.0
        return np.concatenate([generic, normalized_counts])

    def get_feature_dim(self) -> int:
        return 3 + len(self.tracked_names)
