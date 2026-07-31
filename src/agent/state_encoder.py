import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from data.card_lookup import CardLookup
except ImportError:
    CardLookup = None

TYPE_MAP = {
    '{G}': 0, '{R}': 1, '{W}': 2, '{L}': 3, '{P}': 4,
    '{F}': 5, '{D}': 6, '{M}': 7, '{C}': 8, '{N}': 9, '{Y}': 10,
}
NUM_TYPES = 11

def parse_cost(cost_str: str) -> float:
    if not cost_str or cost_str == 'n/a': return 0.0
    return float(cost_str.count('{')) / 5.0

def parse_damage(dmg_str: str) -> float:
    if not dmg_str or dmg_str == 'n/a': return 0.0
    val = ''.join([c for c in str(dmg_str) if c.isdigit()])
    if val:
        return float(val) / 300.0
    return 0.0

class ObservationEncoder:
    """
    Phase 3 encoder.
    Global: 5
    Per Player:
      Active: 52 (slot) + 5 (status) = 57
      Bench: 52 * 5 = 260
      Board: 3
      Total = 320
    Total State: 5 + 320 + 320 = 645
    """
    STATE_DIM = 645

    def __init__(self):
        self.db = CardLookup() if CardLookup is not None else None

    def _get_type_onehot(self, t_str: str) -> np.ndarray:
        vec = np.zeros(NUM_TYPES, dtype=np.float32)
        if t_str and t_str in TYPE_MAP:
            vec[TYPE_MAP[t_str]] = 1.0
        return vec

    def _encode_pokemon_slot(self, slot: dict) -> np.ndarray:
        # Returns 52 dims
        vec = np.zeros(52, dtype=np.float32)
        if not slot:
            return vec
            
        hp = float(slot.get('hp', 0))
        max_hp = float(slot.get('maxHp', max(hp, 1)))
        vec[0] = hp / max(max_hp, 1.0)
        vec[1] = max_hp / 300.0
        
        # Energies
        energies = slot.get('energyCards', [])
        vec[2] = float(len(energies)) / 10.0
        
        # Energy types
        attached_type_vec = np.zeros(NUM_TYPES, dtype=np.float32)
        for e in energies:
            e_id = e.get('id') if isinstance(e, dict) else e
            if self.db and e_id:
                c_rows = self.db.get_card(e_id)
                if c_rows:
                    t = c_rows[0].type
                    if t in TYPE_MAP:
                        attached_type_vec[TYPE_MAP[t]] += 1.0
        vec[3:14] = attached_type_vec / 10.0 # normalize
        
        # Card DB stats
        card_id = slot.get('id')
        if self.db and card_id:
            c_rows = self.db.get_card(card_id)
            if c_rows:
                c = c_rows[0]
                vec[14:25] = self._get_type_onehot(c.type)
                vec[25:36] = self._get_type_onehot(c.weakness)
                vec[36:47] = self._get_type_onehot(c.resistance)
                try:
                    retreat = float(c.retreat) if str(c.retreat).isdigit() else 0.0
                except:
                    retreat = 0.0
                vec[47] = retreat / 4.0
                
                # Moves
                vec[48] = parse_cost(c.cost)
                vec[49] = parse_damage(c.damage)
                if len(c_rows) > 1:
                    c2 = c_rows[1]
                    vec[50] = parse_cost(c2.cost)
                    vec[51] = parse_damage(c2.damage)
                    
        return vec

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

        # Global (5)
        select_data = obs_dict.get('select') or {}
        n_options = float(len(select_data.get('option', [])))
        global_feats = np.array([
            float(curr.get('turn', 0)) / 50.0,
            float(curr.get('supporterPlayed', False)),
            float(curr.get('energyAttached', False)),
            float(curr.get('retreated', False)),
            n_options / 10.0,
        ], dtype=np.float32)

        # Player encoding func
        def encode_player(p: dict) -> np.ndarray:
            active_list = p.get('active', [])
            active_slot = active_list[0] if active_list else {}
            active_feats = self._encode_pokemon_slot(active_slot)
            
            status_feats = np.array([
                float(p.get('poisoned', False)),
                float(p.get('burned', False)),
                float(p.get('asleep', False)),
                float(p.get('paralyzed', False)),
                float(p.get('confused', False)),
            ], dtype=np.float32)
            
            bench_raw = p.get('bench', [])
            bench_feats = np.zeros(5 * 52, dtype=np.float32)
            for i in range(5):
                if i < len(bench_raw):
                    bench_feats[i*52:(i+1)*52] = self._encode_pokemon_slot(bench_raw[i])
                    
            board_feats = np.array([
                float(len(p.get('prize', []))) / 6.0,
                float(p.get('handCount', 0)) / 10.0,
                float(p.get('deckCount', 0)) / 60.0,
            ], dtype=np.float32)
            
            return np.concatenate([active_feats, status_feats, bench_feats, board_feats])

        me_encoded = encode_player(p_me)    # 57 + 260 + 3 = 320
        opp_encoded = encode_player(p_opp)  # 320
        
        state = np.concatenate([global_feats, me_encoded, opp_encoded])
        assert len(state) == self.STATE_DIM, f"State dim mismatch: {len(state)} vs {self.STATE_DIM}"
        return state.astype(np.float32)

    def get_state_dim(self) -> int:
        return self.STATE_DIM
