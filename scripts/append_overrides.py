append_text = r'''

# ==================== META OVERRIDE LAYERS (ported from v15) ====================

# Static public deck signatures for the two most common meta archetypes.
# These activate as soon as the matching Pokemon line is visible on opponent side.
_GRIMMSNARL_IDS = [7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,
                   646,646,646,646,647,647,647,648,648,648,860,860,
                   1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,
                   1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,
                   1227,1227,1227,1227,1231,1259,1259,1259,1259]
_DRAGAPULT_IDS  = [119,119,119,119,120,120,120,121,121,121,
                   131,131,131,131,132,132,133,133,
                   1079,1079,1079,1079,1086,1086,1086,1086,1097,1097,1097,
                   1152,1152,1152,1152,1182,1182,1182,1227,1227,1227,
                   1231,1231,1247,1159,5,5,5,5,11,11,11,11,13,13,
                   1161,1161,1184,1225,1225,17,17]
_GRIMMSNARL_LINE = {646, 647, 648}
_DRAGAPULT_LINE  = {119, 120, 121}

_STATIC_TEMPLATES = [
    ("grimmsnarl_static", _pokemon_ids_from_counter(Counter(_GRIMMSNARL_IDS)),
     Counter(_GRIMMSNARL_IDS), _GRIMMSNARL_IDS),
    ("dragapult_static",  _pokemon_ids_from_counter(Counter(_DRAGAPULT_IDS)),
     Counter(_DRAGAPULT_IDS), _DRAGAPULT_IDS),
]

_TEAM_ROCKET_ENERGY_ID = 15
_ENHANCED_HAMMER_ID = 1081


def _op_has_energy_id(obs_dict, energy_id):
    try:
        obs = to_observation_class(obs_dict)
        state = obs.current
        if state is None: return False
        op = state.players[1 - state.yourIndex]
        for p in (op.active or []) + (op.bench or []):
            if p is not None:
                for e in getattr(p, "energyCards", []):
                    if e.id == energy_id: return True
    except Exception: pass
    return False


def _op_has_line(obs_dict, card_ids):
    try:
        obs = to_observation_class(obs_dict)
        state = obs.current
        if state is None: return False
        op = state.players[1 - state.yourIndex]
        cards = list(op.active or []) + list(op.bench or []) + list(op.discard or [])
        return any(c is not None and c.id in card_ids for c in cards)
    except Exception: return False


_base_lucario_agent = agent


def agent(obs_dict, configuration=None):
    """
    Layered entry point:
    1. Activate static Grimmsnarl/Dragapult templates when those lines are visible.
    2. Run base Lucario heuristic + 2-ply search.
    3. If opponent has Team Rocket Energy, boost Enhanced Hammer to top priority.
    """
    global _TEMPLATE_SIG

    # Activate static archetype templates based on visible opponent Pokemon
    extras = []
    if _op_has_line(obs_dict, _GRIMMSNARL_LINE):
        extras += [t for t in _STATIC_TEMPLATES if "grimmsnarl" in t[0]]
    if _op_has_line(obs_dict, _DRAGAPULT_LINE):
        extras += [t for t in _STATIC_TEMPLATES if "dragapult" in t[0]]
    _TEMPLATE_SIG = extras + [t for t in _TEMPLATE_SIG if not any(
        t[0] == e[0] for e in extras)]

    result = _base_lucario_agent(obs_dict)

    # Team Rocket Energy override: re-rank if Enhanced Hammer should be played
    try:
        if (_op_has_energy_id(obs_dict, _TEAM_ROCKET_ENERGY_ID)
                and isinstance(obs_dict, dict) and obs_dict.get("select")):
            obs = to_observation_class(obs_dict)
            if obs.select is not None and obs.select.context == SelectContext.MAIN:
                policy = LucarioPolicy(obs)
                _, scores = policy.rank_and_scores()
                my_idx = obs.current.yourIndex
                for idx, opt in enumerate(obs.select.option):
                    if opt.type == OptionType.PLAY:
                        card = get_card(obs, AreaType.HAND, opt.index, my_idx)
                        if card is not None and card.id == _ENHANCED_HAMMER_ID:
                            scores[idx] += 20_000
                n = len(obs.select.option)
                ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
                result = ranked[:obs.select.maxCount]
    except Exception: pass

    return result
'''

with open("src/agent/rule_based_lucario.py", "a", encoding="utf-8") as f:
    f.write(append_text)

lines = open("src/agent/rule_based_lucario.py", encoding="utf-8").read().count("\n")
print(f"Done. File now has {lines} lines.")
