import os
import csv

# ─── Load Card Database ───────────────────────────────────────────────────────
CARD_DB = {}
try:
    _csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "EN_Card_Data.csv")
    with open(_csv_path, encoding="utf-8") as _f:
        _r = csv.DictReader(_f)
        for _row in _r:
            _cid = int(_row["Card ID"])
            _hp = int(_row["HP"]) if _row["HP"] and _row["HP"].isdigit() else 0
            CARD_DB[_cid] = {
                "name": _row["Card Name"],
                "hp": _hp,
                "type": _row["Type"],
                "category": _row["Category"]
            }
except Exception as e:
    print(f"Generic Agent: Failed to load CARD_DB: {e}")

# ─── Option type constants from cabt API ─────────────────────────────────────
OPT_ATTACK  = 13
OPT_EVOLVE  = 9
OPT_PLAY    = 7
OPT_ATTACH  = 8
OPT_ABILITY = 10
OPT_RETREAT = 12
OPT_END     = 14

SELECT_MAIN   = 0
SELECT_CARD   = 1
SELECT_YES_NO = 9

# Context codes
CTX_SETUP_ACTIVE   = 1
CTX_SETUP_BENCH    = 2
CTX_SWITCH_IN      = 7
CTX_MAIN           = 0


# 10=Must play, 1=Don't play unless forced
PRIORITY_TRAINERS = {
    # Search / Setup
    "Buddy-Buddy Poffin": 10,
    "Nest Ball": 10,
    "Ultra Ball": 9,
    "Evo Incense": 9,
    "VIP Pass": 9,
    "Level Ball": 8,
    "Quick Ball": 8,
    
    # Draw Support
    "Professor's Research": 8,
    "Iono": 8,
    "Colress's Experiment": 8,
    "Carmine": 7,
    "Kieran": 7,
    
    # Gust / Disruption
    "Boss's Orders": 9,
    "Counter Catcher": 8,
    "Prime Catcher": 9,
    
    # Tools / Recovery
    "Super Rod": 7,
    "Rescue Board": 6,
    "Earthen Vessel": 8,
    "Energy Switch": 7,
    
    # Rare / Anti-combo
    "Hand Trimmer": 1 # Rarely play
}

def _get_players(obs_dict: dict):
    curr = obs_dict.get("current", {}) or {}
    your_idx = curr.get("yourIndex", 0)
    players = curr.get("players", [{}, {}])
    if len(players) < 2:
        return {}, {}
    opp_idx = 1 - your_idx
    me  = players[your_idx] if your_idx  < len(players) else {}
    opp = players[opp_idx]  if opp_idx   < len(players) else {}
    return me, opp

def _find_options_of_type(options: list, opt_type: int) -> list[tuple[int, dict]]:
    return [(i, opt) for i, opt in enumerate(options) if isinstance(opt, dict) and opt.get("type") == opt_type]

def _pick_setup_active(options: list, me: dict) -> int | None:
    active_opts = _find_options_of_type(options, 1)
    if not active_opts: return None
    
    my_hand = me.get("hand") or []
    best_idx = active_opts[0][0]
    best_hp = -1
    
    # Advanced Setup Active: Pick the Basic Pokemon with the highest HP
    for opt_idx, opt in active_opts:
        hand_idx = opt.get("index")
        if hand_idx is not None and hand_idx < len(my_hand):
            card_id = my_hand[hand_idx].get("id", 0)
            db_card = CARD_DB.get(card_id, {})
            hp = db_card.get("hp", 0)
            if hp > best_hp:
                best_hp = hp
                best_idx = opt_idx
                
    return best_idx

def _pick_setup_bench(options: list, me: dict) -> list[int]:
    bench_opts = _find_options_of_type(options, 2)
    if not bench_opts: return []
    
    my_hand = me.get("hand") or []
    scored_opts = []
    
    # Advanced Setup Bench: Sort by HP (highest first)
    for opt_idx, opt in bench_opts:
        hand_idx = opt.get("index")
        if hand_idx is not None and hand_idx < len(my_hand):
            card_id = my_hand[hand_idx].get("id", 0)
            db_card = CARD_DB.get(card_id, {})
            hp = db_card.get("hp", 0)
            scored_opts.append((opt_idx, hp))
            
    # If hand is hidden, just pick all options (like before)
    if not scored_opts:
        return [idx for idx, _ in bench_opts]
        
    scored_opts.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored_opts]

def _pick_play_card(options: list, me: dict, opp: dict) -> int | None:
    play_opts = _find_options_of_type(options, OPT_PLAY)
    if not play_opts: return None
    
    my_hand = me.get("hand") or []
    valid_opts = []
    for opt_idx_in_list, opt in play_opts:
        hand_idx = opt.get("index")
        if hand_idx is not None and hand_idx < len(my_hand):
            card_id = my_hand[hand_idx].get("id", 0)
            db_card = CARD_DB.get(card_id, {})
            card_name = db_card.get("name", "")
            
            # Map name to priority!
            prio = PRIORITY_TRAINERS.get(card_name, 5)
            if prio > 1: # Don't play 1-priority cards unless forced
                valid_opts.append((opt_idx_in_list, prio))
                
    if valid_opts:
        valid_opts.sort(key=lambda x: x[1], reverse=True)
        return valid_opts[0][0]
    
    # Fallback if hand is hidden or no valid priorities found
    return play_opts[0][0]

def _pick_ability(options: list, me: dict) -> int | None:
    ability_opts = _find_options_of_type(options, OPT_ABILITY)
    if not ability_opts: return None
    return ability_opts[0][0]

def _pick_evolve(options: list, me: dict) -> int | None:
    evolve_opts = _find_options_of_type(options, OPT_EVOLVE)
    if evolve_opts: return evolve_opts[0][0]
    return None

def _pick_attach(options: list, me: dict, opp: dict) -> int | None:
    attach_opts = _find_options_of_type(options, OPT_ATTACH)
    if not attach_opts: return None
    
    # Advanced Attach: Attach to Active if it needs it, else highest maxHp Bench!
    best_idx = attach_opts[0][0]
    best_priority = -1
    
    active = me.get("active", [])
    bench = me.get("bench", [])
    
    for opt_idx, opt in attach_opts:
        area = opt.get("inPlayArea")
        play_idx = opt.get("inPlayIndex", 0)
        
        priority = 0
        if area == 4: # Active
            if active:
                priority = active[0].get("maxHp", 100) + 1000 # Active gets massive bonus
        elif area == 5: # Bench
            if play_idx < len(bench):
                priority = bench[play_idx].get("maxHp", 0)
                
        if priority > best_priority:
            best_priority = priority
            best_idx = opt_idx
            
    return best_idx

def _pick_retreat(options: list, me: dict) -> int | None:
    retreat_opts = _find_options_of_type(options, OPT_RETREAT)
    if retreat_opts:
        active = me.get("active", [])
        if active:
            hp = active[0].get("hp", 100)
            max_hp = active[0].get("maxHp", 100)
            # Advanced Retreat: Only retreat if dying and we have something else!
            if hp <= 50 and hp < max_hp * 0.5:
                return retreat_opts[0][0]
    return None

def _pick_attack(options: list, me: dict, opp: dict, obs_dict: dict) -> int | None:
    attack_opts = _find_options_of_type(options, OPT_ATTACK)
    if not attack_opts: return None
    # Advanced Attack: The engine usually lists the weakest basic attack first (index 0) 
    # and the strongest ultimate attack last (index 1 or 2).
    # Since string names are stripped in Fast C++, we aggressively pick the highest index attack!
    return attack_opts[-1][0]

def _handle_card_select(obs_dict: dict, options: list) -> list[int]:
    select_data = obs_dict.get("select", {})
    ctx = select_data.get("context")
    min_count = select_data.get("minCount", 1)
    opts_len = len(options)
    
    if ctx in [CTX_SWITCH_IN, 7, "SWITCH_IN", "TO_ACTIVE"]:
        if min_count > 1:
            return list(range(min(opts_len, min_count)))
            
        me, opp = _get_players(obs_dict)
        bench = me.get("bench", [])
        # Advanced Switch: Pick the bench pokemon with the most HP!
        if bench and opts_len <= len(bench):
            best_idx = 0
            best_hp = -1
            for opt_idx, opt in enumerate(options):
                b_idx = opt.get("index", opt_idx)
                if b_idx < len(bench):
                    b_hp = bench[b_idx].get("hp", 0)
                    if b_hp > best_hp:
                        best_hp = b_hp
                        best_idx = opt_idx
            return [best_idx]
        return [0]
        
    if min_count > 1 and opts_len > 0:
        return list(range(min(opts_len, min_count)))
    elif min_count == 1 and opts_len >= 1:
        return [0]
    return []

def rule_based_generic_agent(obs_dict: dict) -> list[int]:
    try:
        step = obs_dict.get("step", 1) if obs_dict else 0
        if step == 0:
            return []

        select_data = obs_dict.get("select", {})
        select_type = select_data.get("type", SELECT_MAIN)
        select_ctx = select_data.get("context", CTX_MAIN)
        options = select_data.get("option", [])

        me, opp = _get_players(obs_dict)

        if select_type == SELECT_YES_NO:
            return [0]

        if not options:
            return []

        if select_type == SELECT_CARD:
            if select_ctx in {CTX_SETUP_ACTIVE, 1, "SETUP_ACTIVE"}:
                idx = _pick_setup_active(options, me)
                return [idx] if idx is not None else ([0] if options else [])
            if select_ctx in {CTX_SETUP_BENCH, 2, "SETUP_BENCH"}:
                return _pick_setup_bench(options, me)
            return _handle_card_select(obs_dict, options)
            
        elif select_type == SELECT_MAIN:
            if select_ctx == CTX_MAIN:
                # 1. Always Evolve First
                ev_idx = _pick_evolve(options, me)
                if ev_idx is not None: return [ev_idx]
                
                # 2. Play Generic Setup Trainers
                play_idx = _pick_play_card(options, me, opp)
                if play_idx is not None: return [play_idx]
                
                # 3. Use Draw/Setup Abilities
                ab_idx = _pick_ability(options, me)
                if ab_idx is not None: return [ab_idx]
                
                # 4. Attach Energy
                at_idx = _pick_attach(options, me)
                if at_idx is not None: return [at_idx]
                
                # 5. Attack!
                att_idx = _pick_attack(options, me, opp, obs_dict)
                if att_idx is not None: return [att_idx]
                
                # 6. End turn if nothing left
                end_opts = _find_options_of_type(options, OPT_END)
                if end_opts: return [end_opts[0][0]]
                
        # Fallback for ANY unknown select_type or context
        min_count = select_data.get("minCount", 1)
        opts_len = len(options)
        if min_count > 1 and opts_len > 0:
            return list(range(min(opts_len, min_count)))
        elif opts_len > 0:
            return [0]
        return []
        
    except Exception as e:
        # Extreme fallback to prevent valid option crashes
        select_data = obs_dict.get("select", {}) if obs_dict else {}
        min_count = select_data.get("minCount", 1)
        options = select_data.get("option", [])
        if min_count > 1 and len(options) > 0:
            return list(range(min(len(options), min_count)))
        elif len(options) > 0:
            return [0]
        return []
