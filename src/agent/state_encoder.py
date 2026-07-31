import numpy as np

# Pokémon type to index mapping for one-hot encoding
TYPE_MAP = {
    '{G}': 0, '{R}': 1, '{W}': 2, '{L}': 3, '{P}': 4,
    '{F}': 5, '{D}': 6, '{M}': 7, '{C}': 8, '{N}': 9, '{Y}': 10,
}
NUM_TYPES = 11

class ObservationEncoder:
    """
    Encodes the cabt JSON observation dict into a flat float32 vector.

    Phase 3 encoding: ~128-dim
    - Global (5):      turn, supporter_played, energy_attached, retreated, n_legal_options
    - My Active (13):  hp_norm, n_energies, 11-type one-hot (sum over attached)
    - My Bench x5 (13 each = 65): same as active, per slot
    - My Board (3):    prizes_remaining, hand_count, deck_count
    - Opp Active (13): same
    - Opp Bench x5 (13 each = 65): same
    - Opp Board (3):   same
    Total: 5 + 13 + 65 + 3 + 13 + 65 + 3 = 167 dims
    """

    STATE_DIM = 167

    def __init__(self):
        pass

    def _encode_pokemon_slot(self, slot: dict) -> np.ndarray:
        """Encode a single Pokémon slot into 13 floats."""
        if not slot:
            return np.zeros(13, dtype=np.float32)

        # HP fraction (normalize by 300 — max realistic HP)
        hp = slot.get('hp', 0)
        max_hp = slot.get('maxHp', max(hp, 1))
        hp_frac = float(hp) / float(max(max_hp, 1))

        # Number of attached energies
        energies = slot.get('energy', [])
        n_energies = float(len(energies)) / 5.0  # normalize by 5

        # Type one-hot (sum over all attached energy types)
        type_vec = np.zeros(NUM_TYPES, dtype=np.float32)
        for e in energies:
            etype = e.get('type', '')
            idx = TYPE_MAP.get(etype, -1)
            if idx >= 0:
                type_vec[idx] += 1.0

        return np.array([hp_frac, n_energies] + type_vec.tolist(), dtype=np.float32)

    def encode(self, obs_dict: dict) -> np.ndarray:
        if obs_dict is None or obs_dict.get('current') is None:
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        curr = obs_dict['current']
        your_idx = curr.get('yourIndex', 0)
        opp_idx = 1 - your_idx

        players = curr.get('players', [{}, {}])
        if len(players) < 2:
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        p_me = players[your_idx] if your_idx < len(players) else {}
        p_opp = players[opp_idx] if opp_idx < len(players) else {}

        # --- Global (5) ---
        select_data = obs_dict.get('select') or {}
        n_options = float(len(select_data.get('option', [])))
        global_feats = np.array([
            float(curr.get('turn', 0)) / 50.0,           # normalize by 50 turns
            float(curr.get('supporterPlayed', False)),
            float(curr.get('energyAttached', False)),
            float(curr.get('retreated', False)),
            n_options / 10.0,                             # normalize by 10
        ], dtype=np.float32)

        # --- My Active (13) ---
        my_active_list = p_me.get('active', [{}])
        my_active = self._encode_pokemon_slot(my_active_list[0] if my_active_list else {})

        # --- My Bench x5 (65) ---
        my_bench_raw = p_me.get('bench', [])
        my_bench_feats = np.zeros(5 * 13, dtype=np.float32)
        for i in range(5):
            if i < len(my_bench_raw):
                my_bench_feats[i*13:(i+1)*13] = self._encode_pokemon_slot(my_bench_raw[i])

        # --- My Board (3) ---
        my_board = np.array([
            float(len(p_me.get('prize', []))) / 6.0,
            float(p_me.get('handCount', 0)) / 10.0,
            float(p_me.get('deckCount', 0)) / 60.0,
        ], dtype=np.float32)

        # --- Opp Active (13) ---
        opp_active_list = p_opp.get('active', [{}])
        opp_active = self._encode_pokemon_slot(opp_active_list[0] if opp_active_list else {})

        # --- Opp Bench x5 (65) ---
        opp_bench_raw = p_opp.get('bench', [])
        opp_bench_feats = np.zeros(5 * 13, dtype=np.float32)
        for i in range(5):
            if i < len(opp_bench_raw):
                opp_bench_feats[i*13:(i+1)*13] = self._encode_pokemon_slot(opp_bench_raw[i])

        # --- Opp Board (3) ---
        opp_board = np.array([
            float(len(p_opp.get('prize', []))) / 6.0,
            float(p_opp.get('handCount', 0)) / 10.0,
            float(p_opp.get('deckCount', 0)) / 60.0,
        ], dtype=np.float32)

        state = np.concatenate([
            global_feats,    # 5
            my_active,       # 13
            my_bench_feats,  # 65
            my_board,        # 3
            opp_active,      # 13
            opp_bench_feats, # 65
            opp_board,       # 3
        ])

        assert len(state) == self.STATE_DIM, f"State dim mismatch: {len(state)} vs {self.STATE_DIM}"
        return state.astype(np.float32)

    def get_state_dim(self) -> int:
        return self.STATE_DIM
