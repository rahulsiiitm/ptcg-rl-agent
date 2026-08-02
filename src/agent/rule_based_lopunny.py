import os
import sys
import random

# ─── Card ID Constants ────────────────────────────────────────────────────────
BUNEARY_ID = 848
MEGA_LOPUNNY_EX_ID = 849
DUNSPARCE_ID = 305
DUDUNSPARCE_ID = 66
SNORUNT_ID = 860
MEGA_FROSLASS_EX_ID = 861
FAN_ROTOM_ID = 174

BUDDY_BUDDY_ID = 1086
ULTRA_BALL_ID = 1121
POKEGEAR_ID = 1122
POKE_PAD_ID = 1152
AIR_BALLOON_ID = 1174
BOSS_ORDERS_ID = 1182
HILDA_ID = 1225
LILLIE_ID = 1227
WALLY_ID = 1229
HAND_TRIMMER_ID = 1087
BATTLE_CAGE_ID = 1264
WATER_ENERGY_ID = 3
MIST_ENERGY_ID = 11

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
CTX_TO_HAND        = 4
CTX_DISCARD        = 5
CTX_EFFECT_TARGET  = 6
CTX_SWITCH_IN      = 7
CTX_BENCH_PLACE    = 8
CTX_DISCARD_ENERGY = 10
CTX_SWITCH_SELF    = 11
CTX_MAIN           = 0

def _read_deck() -> list[int]:
    for path in [
        "deck.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ]:
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            return [int(x) for x in lines[:60]]
    return []

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

def _active_id(player: dict) -> int | None:
    active_list = player.get("active", [])
    if not active_list: return None
    slot = active_list[0]
    return slot.get("id") if isinstance(slot, dict) else None

def _bench_ids(player: dict) -> list[int]:
    return [s.get("id") for s in player.get("bench", []) if isinstance(s, dict) and s.get("id") is not None]

def _find_options_of_type(options: list, opt_type: int) -> list[tuple[int, dict]]:
    return [(i, opt) for i, opt in enumerate(options) if isinstance(opt, dict) and opt.get("type") == opt_type]

def _pick_setup_active(options: list, me: dict) -> int | None:
    active_opts = _find_options_of_type(options, 1)
    if not active_opts: return None
    priority = {FAN_ROTOM_ID: 4, DUNSPARCE_ID: 3, SNORUNT_ID: 2, BUNEARY_ID: 1}
    active_opts.sort(key=lambda x: priority.get(x[1].get("card", {}).get("id", 0), 0), reverse=True)
    return active_opts[0][0]

def _pick_setup_bench(options: list) -> list[int]:
    return [i for i, opt in enumerate(options) if isinstance(opt, dict) and opt.get("type") == 2]

def _pick_play_card(options: list, me: dict, opp: dict) -> int | None:
    play_opts = _find_options_of_type(options, OPT_PLAY)
    if not play_opts: return None
    
    priority = {
        BUDDY_BUDDY_ID: 10,
        ULTRA_BALL_ID: 9,
        POKE_PAD_ID: 8,
        POKEGEAR_ID: 7,
        HILDA_ID: 6,
        LILLIE_ID: 5,
        BATTLE_CAGE_ID: 4,
        AIR_BALLOON_ID: 3,
        BOSS_ORDERS_ID: 2,
        WALLY_ID: 1,
        HAND_TRIMMER_ID: 0
    }
    
    # Don't play Hand Trimmer unless we have a massive hand
    my_hand = len(me.get("hand", []))
    
    valid_opts = []
    for idx, opt in play_opts:
        cid = opt.get("card", {}).get("id", 0)
        if cid == HAND_TRIMMER_ID and my_hand < 8:
            continue
        valid_opts.append((idx, priority.get(cid, -1)))
        
    if not valid_opts:
        return None
        
    valid_opts.sort(key=lambda x: x[1], reverse=True)
    return valid_opts[0][0]

def _pick_ability(options: list, me: dict) -> int | None:
    ability_opts = _find_options_of_type(options, OPT_ABILITY)
    if not ability_opts: return None
    
    priority = {DUDUNSPARCE_ID: 10, FAN_ROTOM_ID: 5}
    ability_opts.sort(key=lambda x: priority.get(x[1].get("card", {}).get("id", 0), 0), reverse=True)
    
    for idx, opt in ability_opts:
        cid = opt.get("card", {}).get("id")
        if cid == DUDUNSPARCE_ID:
            # Only use if it won't lose us the game
            bench = _bench_ids(me)
            active = _active_id(me)
            if len(bench) > 0 or (len(bench) == 0 and active != DUDUNSPARCE_ID):
                return idx
        else:
            return idx
    return None

def _pick_evolve(options: list, me: dict) -> int | None:
    evolve_opts = _find_options_of_type(options, OPT_EVOLVE)
    if evolve_opts: return evolve_opts[0][0]
    return None

def _pick_attach(options: list, me: dict) -> int | None:
    attach_opts = _find_options_of_type(options, OPT_ATTACH)
    if attach_opts:
        # Prefer attaching Mist energy to Mega Lopunny / Froslass
        priority_targets = {MEGA_LOPUNNY_EX_ID: 10, MEGA_FROSLASS_EX_ID: 9, DUNSPARCE_ID: 5}
        # In this simplistic logic, just return the first available, or ideally filter
        return attach_opts[0][0]
    return None

def _pick_retreat(options: list, me: dict) -> int | None:
    retreat_opts = _find_options_of_type(options, OPT_RETREAT)
    if retreat_opts: return retreat_opts[0][0]
    return None

def _pick_attack(options: list, me: dict, opp: dict, obs_dict: dict) -> int | None:
    attack_opts = _find_options_of_type(options, OPT_ATTACK)
    if not attack_opts: return None
    # Prioritize Resentful Refrain or Gale Thrust
    priority = {"Resentful Refrain": 10, "Gale Thrust": 9, "Absolute Snow": 5, "Chilly": 1}
    attack_opts.sort(key=lambda x: priority.get(x[1].get("move", {}).get("name", ""), 0), reverse=True)
    return attack_opts[0][0]

def _handle_card_select(obs_dict: dict, options: list) -> list[int]:
    select_data = obs_dict.get("select", {})
    ctx = select_data.get("context")
    min_count = select_data.get("minCount", 1)
    opts_len = len(options)
    
    if ctx in [CTX_SWITCH_IN, 7, "SWITCH_IN", "TO_ACTIVE"]:
        priority = {MEGA_LOPUNNY_EX_ID: 10, MEGA_FROSLASS_EX_ID: 9, DUDUNSPARCE_ID: 5, DUNSPARCE_ID: 4, FAN_ROTOM_ID: 3}
        sorted_opts = sorted(enumerate(options), key=lambda x: priority.get(x[1].get("card", {}).get("id", 0), 0), reverse=True)
        return [sorted_opts[0][0]]
        
    if min_count > 1 and opts_len >= min_count:
        return list(range(min_count))
    elif min_count == 1 and opts_len >= 1:
        return [0]
    return []

def rule_based_agent(obs_dict: dict) -> list[int]:
    print("Agent called! Step:", obs_dict.get('step', -1), flush=True)
    try:
        step = obs_dict.get("step", 0)
        if step == 0:
            return _read_deck()

        select_data = obs_dict.get("select", {})
        select_type = select_data.get("type", SELECT_MAIN)
        select_ctx = select_data.get("context", CTX_MAIN)
        options = select_data.get("option", [])

        me, opp = _get_players(obs_dict)

        if not options:
            return []

        if select_type == SELECT_CARD:
            if select_ctx in {CTX_SETUP_ACTIVE, 1, "SETUP_ACTIVE"}:
                idx = _pick_setup_active(options, me)
                return [idx] if idx is not None else ([0] if options else [])
            if select_ctx in {CTX_SETUP_BENCH, 2, "SETUP_BENCH"}:
                return _pick_setup_bench(options)
            return _handle_card_select(obs_dict, options)
            
        elif select_type == SELECT_YES_NO:
            return [0]

        elif select_type == SELECT_MAIN:
            if select_ctx == CTX_MAIN:
                ab_idx = _pick_ability(options, me)
                if ab_idx is not None: return [ab_idx]
                
                ev_idx = _pick_evolve(options, me)
                if ev_idx is not None: return [ev_idx]
                
                play_idx = _pick_play_card(options, me, opp)
                if play_idx is not None: return [play_idx]
                
                at_idx = _pick_attach(options, me)
                if at_idx is not None: return [at_idx]
                
                active_id = _active_id(me)
                if active_id not in [MEGA_LOPUNNY_EX_ID, MEGA_FROSLASS_EX_ID]:
                    ret_idx = _pick_retreat(options, me)
                    if ret_idx is not None: return [ret_idx]
                    
                att_idx = _pick_attack(options, me, opp, obs_dict)
                if att_idx is not None: return [att_idx]
                
                end_opts = _find_options_of_type(options, OPT_END)
                if end_opts: return [end_opts[0][0]]
                
            return [0]

        return [0]
    except Exception as e:
        # Fallback to prevent invalid crashes
        return [0]
