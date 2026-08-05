import os, json
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
alak_evo_wm: memetic-tuned alak_evo with parametrized priority weights.
Default WEIGHTS reproduce alak_evo exactly. Override via ./alak_w.json.
"""

# ---- Tunable priority weights (defaults = alak_evo values) ----
WEIGHTS = {
    # PLAY: PokÃ©mon
    "play_pokemon_base": 20000,
    "play_abra_early": 500, "play_abra_need": 200, "play_abra_extra": 50,
    "play_dun_first_early": 400, "play_dun_first_late": 100, "play_dun_second": 50, "play_dun_ex": 30,
    "play_fez": 20080, "play_genesect": 20100, "play_psyduck": 20300, "play_shaymin": 20300, "play_fanrotom": 20250,
    "play_bench_penalty": 5000,
    # PLAY: Trainers
    "poffin_early": 18000, "poffin_fallback": 8000, "poffin_late": 15000,
    "pokepad_early": 17000, "pokepad_need": 14000, "pokepad_ok": 12000,
    "rare_candy": 16000,
    "night_stretcher_mon": 13000, "night_stretcher_energy": 11000,
    "sacred_ash_hi": 13500, "sacred_ash_lo": 11000,
    "hammer_target": 6500, "hammer_any": 5000,
    "wondrous_patch": 8500, "meddling_memo": 6000,
    "boss_kill": 3200, "hilda": 3000, "dawn_emergency": 16500, "dawn": 3100,
    "lillie": 3400, "lana": 3300, "xerosic": 3250, "eri": 3150,
    "nz_ex": 19500, "nz_counter": 7500,
    "cage_counter": 19000, "cage_snipe": 18500,
    "mine_counter": 18800, "jamming_tools": 18900, "jamming_counter": 18700,
    # ATTACH
    "helmet": 7000, "fan_abra": 7200, "fan_genesect": 7100, "balloon": 7300,
    "cape_alak": 9800, "cape_kadabra": 9600, "cape_abra": 7500,
    "energy_retreat": 9500, "energy_abra": 8000,
    "enriching_2nd": 4500, "enriching_1st": 2000,
    "mist_2nd": 4200, "mist_retreat": 9400,
    # EVOLVE / ABILITY / RETREAT / ATTACK
    "evolve_base": 9000,
    "ability_dudun": 30000, "ability_fez": 29000, "ability_fanrotom": 29500, "ability_default": 28000,
    "retreat_kadabra": 2500, "retreat_promote": 2000,
    "attack_base": 1000, "attack_powerful": 500, "attack_psybolt_kill": 600, "attack_psybolt": 100, "attack_teleport": 50,
}
# ---- memetic-tuned overrides (baked, seed for wm4 evo) ----
WEIGHTS.update({"play_pokemon_base": 20000, "play_abra_early": 604, "play_abra_need": 200, "play_abra_extra": 50, "play_dun_first_early": 178, "play_dun_first_late": 70, "play_dun_second": 50, "play_dun_ex": 59, "play_fez": 20080, "play_genesect": 20100, "play_psyduck": 20300, "play_shaymin": 19807, "play_fanrotom": 20250, "play_bench_penalty": 3923, "poffin_early": 18000, "poffin_fallback": 4083, "poffin_late": 15000, "pokepad_early": 17000, "pokepad_need": 14000, "pokepad_ok": 12000, "rare_candy": 16000, "night_stretcher_mon": 13000, "night_stretcher_energy": 11000, "sacred_ash_hi": 13500, "sacred_ash_lo": 11000, "hammer_target": 6500, "hammer_any": 6993, "wondrous_patch": 8500, "meddling_memo": 6000, "boss_kill": 2262, "hilda": 3000, "dawn_emergency": 16500, "dawn": 3100, "lillie": 3400, "lana": 4249, "xerosic": 3250, "eri": 3150, "nz_ex": 19500, "nz_counter": 7500, "cage_counter": 19000, "cage_snipe": 18500, "mine_counter": 18495, "jamming_tools": 18900, "jamming_counter": 18700, "helmet": 7000, "fan_abra": 4301, "fan_genesect": 5611, "balloon": 7300, "cape_alak": 9800, "cape_kadabra": 9600, "cape_abra": 7500, "energy_retreat": 9500, "energy_abra": 8000, "enriching_2nd": 6249, "enriching_1st": 2000, "mist_2nd": 4200, "mist_retreat": 9400, "evolve_base": 5951, "ability_dudun": 30000, "ability_fez": 38066, "ability_fanrotom": 29500, "ability_default": 38085, "retreat_kadabra": 2500, "retreat_promote": 2000, "attack_base": 1000, "attack_powerful": 655, "attack_psybolt_kill": 600, "attack_psybolt": 127, "attack_teleport": 67})
# Load runtime overrides (evo genome / final baked) â€” AFTER seed so they win.
for _p in ("alak_w.json", "./alak_w.json",
           "agents/alak_evo2_weights.json", "alak_evo2_weights.json",
           "/kaggle_simulations/agent/alak_evo2_weights.json",
           "/kaggle_simulations/agent/alak_w.json"):
    if os.path.exists(_p):
        try:
            WEIGHTS.update(json.load(open(_p)))
        except Exception:
            pass
        break
W = WEIGHTS

my_deck = []

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ---- Core ----
Abra = 741
Kadabra = 742
Alakazam = 743
Dunsparce65 = 65
Dunsparce305 = 305
Dudunsparce = 66
Fezandipiti_ex = 140
Genesect = 142
Shaymin = 343
Psyduck = 858
Fan_Rotom = 174
Rare_Candy = 1079
Enhanced_Hammer = 1081
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Meddling_Memo = 1103
Sacred_Ash = 1129
Wondrous_Patch = 1146
Poke_Pad = 1152
Switch = 1123
Lucky_Helmet = 1156
Hero_Cape = 1159
Handheld_Fan = 1161
Air_Balloon = 1174
Boss_Orders = 1182
Lanas_Aid = 1184
Eri = 1186
Xerosic = 1197
Hilda = 1225
Lillie_Det = 1227
Dawn = 1231
Full_Metal_Lab = 1244
Jamming_Tower = 1246
Neutralization_Zone = 1247
Battle_Cage = 1264
Nighttime_Mine = 1266
Basic_Psychic_Energy = 5
Mist_Energy = 11
Enriching_Energy = 13
Telepath_Psychic_Energy = 19
Rock_Fighting_Energy = 20

OUR_STADIUMS = {Neutralization_Zone, Battle_Cage, Nighttime_Mine, Jamming_Tower}
DUNSPARCE_IDS = {Dunsparce65, Dunsparce305}
DUNSPARCE_LINE = DUNSPARCE_IDS | {Dudunsparce}
ABRA_LINE = {Abra, Kadabra, Alakazam}
PSYCHIC_ENERGY_IDS = {Basic_Psychic_Energy, Telepath_Psychic_Energy}
TECH_BASICS = {Fezandipiti_ex, Genesect, Shaymin, Psyduck, Fan_Rotom}

Duraludon, Archaludon_ex = 169, 190
Dreepy, Drakloak, Dragapult_ex = 119, 120, 121
Impidimp_G, Morgrem_G, Grimmsnarl_ex = 646, 647, 648
Munkidori = 112
Duskull = 131
Slowpoke_IDs = (162, 327)
Froakie_IDs = (33, 945)
Wellspring_Mask_Ogerpon_ex = 108
N_Darumaka = 257

ATTACK_TELEPORTATION = 1070
ATTACK_SUPER_PSY_BOLT = 1071
ATTACK_POWERFUL_HAND = 1072

pre_turn = 0
ability_used_dudunsparce = False
ability_used_fezandipiti = False


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK: return obs.select.deck[index]
        case AreaType.HAND: return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE: return ps.active[index]
        case AreaType.BENCH: return ps.bench[index]
        case AreaType.PRIZE: return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _: return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12: count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name: count -= 1
    return max(0, count)


def count_special_defense_energies(pokemon):
    return sum(1 for ec in pokemon.energyCards if ec.id in (Mist_Energy, Rock_Fighting_Energy))


def heuristic_scores(obs):
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_count = len(my_state.prize)

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    my_field = []
    for card in my_state.active:
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((0, card))
    for idx, card in enumerate(my_state.bench):
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((idx + 1, card))
    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    abra_line_on_field = sum(field_counts[x] for x in ABRA_LINE)
    dunsparce_on_field = sum(field_counts[x] for x in DUNSPARCE_IDS)
    dunsparce_line_on_field = dunsparce_on_field + field_counts[Dudunsparce]

    op_all_pokemon = [p for p in (op_state.active + op_state.bench) if p is not None]
    op_has_dragapult_line = any(p.id in (Dreepy, Drakloak, Dragapult_ex) for p in op_all_pokemon)
    op_has_grimmsnarl = any(p.id in (Impidimp_G, Morgrem_G, Grimmsnarl_ex, Munkidori) for p in op_all_pokemon)
    op_has_ex = any((cd := card_table.get(p.id)) and (cd.ex or cd.megaEx) for p in op_all_pokemon)
    op_has_duskull = any(p.id == Duskull for p in op_all_pokemon)
    op_has_water_threat = any(
        p.id in Slowpoke_IDs or p.id in Froakie_IDs
        or p.id == Wellspring_Mask_Ogerpon_ex or p.id == N_Darumaka
        for p in op_all_pokemon)
    op_has_tools = any(len(p.tools) > 0 for p in op_all_pokemon)
    op_used_ace_spec = False

    stadium_id = state.stadium[0].id if state.stadium else 0
    our_stadium_up = stadium_id in OUR_STADIUMS

    bench_count = len([b for b in my_state.bench if b])
    bench_max = my_state.benchMax
    bench_free = bench_max - bench_count

    active_pokemon = my_state.active[0] if my_state.active else None
    active_id = active_pokemon.id if active_pokemon else -1
    active_has_psychic = active_pokemon and any(ec.id in PSYCHIC_ENERGY_IDS for ec in active_pokemon.energyCards)

    op_active = op_state.active[0] if op_state.active else None
    op_active_hp = op_active.hp if op_active else 9999

    hand_size = len(my_state.hand) if my_state.hand else my_state.handCount

    def estimate_hand_increase():
        max_inc = 0
        for _, p in my_field:
            if p.id == Abra and hand_counts[Kadabra] > 0: max_inc += 1
            elif p.id == Abra and hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0: max_inc += 1
            elif p.id == Kadabra and hand_counts[Alakazam] > 0: max_inc += 2
            elif p.id in DUNSPARCE_IDS and hand_counts[Dudunsparce] > 0: max_inc += 1
            elif p.id == Dudunsparce and not ability_used_dudunsparce: max_inc += 3
            elif p.id == Fezandipiti_ex and not ability_used_fezandipiti: max_inc += 3
        if hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
            max_inc += 2
        supporter_options = []
        if not state.supporterPlayed:
            if hand_counts[Hilda] > 0: supporter_options.append(1)
            if hand_counts[Dawn] > 0: supporter_options.append(2)
            if hand_counts[Boss_Orders] > 0: supporter_options.append(-1)
        if supporter_options: max_inc += max(supporter_options)
        if hand_counts[Enriching_Energy] > 0 and not state.energyAttached:
            if active_id == Alakazam and active_has_psychic: max_inc += 3
        return 0, max_inc

    _, max_hand_inc = estimate_hand_increase()
    max_hand_size = hand_size + max_hand_inc
    max_damage = max_hand_size * 20

    target_idx = -1; target_pokemon = None; target_use_boss = False
    target_can_kill = False; target_prize_gain = 0; target_hammer_needed = 0
    use_kadabra_finish = False

    if state.turn >= 2 and op_active is not None:
        if op_active_hp <= 30 and (field_counts[Kadabra] >= 1 or active_id == Kadabra):
            target_idx = 0; target_pokemon = op_active; target_can_kill = True
            target_prize_gain = prize_count(op_active); use_kadabra_finish = True
        else:
            all_op = [(0, op_active)] + [(bi + 1, bp) for bi, bp in enumerate(op_state.bench) if bp]
            candidates = []
            for oi, pkmn in all_op:
                pz = prize_count(pkmn)
                sp_e = count_special_defense_energies(pkmn)
                eff_max_dmg = max_damage; hm_need = 0
                if sp_e > 0:
                    if hand_counts[Enhanced_Hammer] >= sp_e:
                        hm_need = sp_e
                        eff_max_dmg = (max_hand_size - hm_need) * 20
                    else:
                        eff_max_dmg = 0
                ck = pkmn.hp <= eff_max_dmg and eff_max_dmg > 0
                candidates.append((oi, pkmn, pz, ck, hm_need))
            win_cands = [c for c in candidates if c[3] and my_prize_count <= c[2]]
            if win_cands:
                best = min(win_cands, key=lambda x: (0 if x[0] == 0 else 1, -x[1].hp))
                target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                target_use_boss = target_idx != 0
            else:
                killable = [c for c in candidates if c[3]]
                if killable:
                    best = max(killable, key=lambda x: (x[2], x[1].hp))
                    target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                    target_use_boss = target_idx != 0
                else:
                    target_idx = 0; target_pokemon = op_active

    need_dudunsparce_draw = False
    if target_pokemon is not None and target_can_kill:
        if (hand_size - target_hammer_needed) * 20 < target_pokemon.hp:
            need_dudunsparce_draw = True
    if not target_can_kill and any(prize_count(p) >= 2 and p.hp > hand_size * 20 for p in op_all_pokemon):
        need_dudunsparce_draw = True

    fez_contrib = 0
    if field_counts[Fezandipiti_ex] >= 1 and not ability_used_fezandipiti: fez_contrib = 3
    elif hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0: fez_contrib = 2
    need_fez = False
    if target_pokemon is not None and target_can_kill and fez_contrib > 0:
        if (max_hand_size - fez_contrib - target_hammer_needed) * 20 < target_pokemon.hp:
            need_fez = True

    need_retreat_energy = False
    if active_pokemon is not None and state.turn >= 2:
        active_is_attacker = (active_id == Alakazam and active_has_psychic) or (use_kadabra_finish and active_id == Kadabra)
        if not active_is_attacker:
            has_bench_attacker = ((use_kadabra_finish and field_counts[Kadabra] >= 1 and active_id != Kadabra)
                                  or (field_counts[Alakazam] >= 1 and active_id != Alakazam)
                                  or (field_counts[Kadabra] >= 1 and active_id != Kadabra))
            if has_bench_attacker:
                rc = card_table[active_pokemon.id].retreatCost
                if any(t.id == Air_Balloon for t in active_pokemon.tools): rc = max(0, rc - 2)
                if len(active_pokemon.energies) < rc:
                    need_retreat_energy = True

    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain
    deck_count = my_state.deckCount
    safe_draws = deck_count - my_prize_count - 1 if not can_win_this_turn else 999

    max_op_hp = max((p.hp for p in op_all_pokemon), default=0)
    overdraw = False
    has_attack_option = any(o.type == OptionType.ATTACK for o in select.option)
    attack_locked = (context == SelectContext.MAIN and active_id == Alakazam
                     and active_has_psychic and not has_attack_option)
    bench_ready_alak = any(p.id == Alakazam and any(e.id in PSYCHIC_ENERGY_IDS for e in p.energyCards)
                           for fi, p in my_field if fi > 0)

    bench_abra_no_energy = any(p.id in ABRA_LINE and len(p.energyCards) == 0
                               for fi, p in my_field if fi > 0)

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = -1 if context == SelectContext.IS_FIRST else 1
        elif o.type == OptionType.NO:
            score = 5 if context == SelectContext.IS_FIRST else 0

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None: scores.append(0); continue
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                if o.playerIndex == my_index:
                    if card.id == Alakazam: score += 100 + energy_count * 10
                    elif card.id == Kadabra: score += 90 if op_active_hp <= 30 else 30
                    elif card.id == Abra: score += 10
                    elif card.id in DUNSPARCE_LINE: score += 5
                    else: score += 1
                else:
                    if target_use_boss and target_pokemon is not None and o.index == target_idx - 1:
                        score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id in DUNSPARCE_IDS: score = 12
                elif card.id == Abra: score = 10
                elif card.id in TECH_BASICS: score = 2

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == Abra:
                    score = 200 if abra_line_on_field == 0 else 100 + (3 - abra_line_on_field) * 10
                elif card.id in DUNSPARCE_IDS:
                    score = 150 if dunsparce_line_on_field == 0 else 50

            elif context == SelectContext.TO_HAND:
                score = 200 - hand_counts.get(card.id, 0) * 50
                bench_emergency = len(my_field) <= 1
                if card.id == Dudunsparce:
                    score += 80 if (dunsparce_on_field >= 1 and field_counts[Dudunsparce] == 0
                                    and not bench_emergency) else -50
                elif card.id == Kadabra:
                    score += 70 if (field_counts[Abra] >= 1 and not bench_emergency) else -20
                elif card.id == Alakazam:
                    score += 60 if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra:
                    score += 200 if bench_emergency else (50 if abra_line_on_field < 3 else -50)
                elif card.id in DUNSPARCE_IDS:
                    score += 180 if bench_emergency else (40 if dunsparce_line_on_field < 2 else -50)
                elif card.id in PSYCHIC_ENERGY_IDS:
                    score += 30 if not state.energyAttached else -10
                elif card.id == Rare_Candy:
                    score += 40 if field_counts[Abra] >= 1 else -10
                elif card.id == Neutralization_Zone:
                    score += 65 if op_has_ex else 0
                elif card.id in OUR_STADIUMS:
                    score += 25 if not our_stadium_up else -30
                elif card.id == Enriching_Energy:
                    score += 20

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if need_retreat_energy and o.area == AreaType.ACTIVE: score = 150
                    elif len(card.energyCards) >= 1: score = -1
                    elif card.id in ABRA_LINE:
                        score = 100 + {Alakazam: 20, Kadabra: 10, Abra: 0}.get(card.id, 0)
                        if o.area == AreaType.ACTIVE: score += 5
                    elif card.id in DUNSPARCE_LINE: score = 50
                    else: score = 10

            elif context == SelectContext.TO_BENCH:
                if card.id == Abra: score = 100 - abra_line_on_field * 5
                elif card.id in DUNSPARCE_IDS: score = 80 - dunsparce_line_on_field * 5
                elif card.id == Psyduck: score = 60 if op_has_duskull else -1
                elif card.id == Shaymin: score = 55 if (op_has_water_threat or op_has_grimmsnarl) else -1

            elif context == SelectContext.TO_DECK:
                if card.id in ABRA_LINE: score = 100
                elif card.id in DUNSPARCE_LINE: score = 50
                else: score = 10

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]

            if data.cardType == CardType.POKEMON:
                score = W["play_pokemon_base"]
                is_early = state.turn <= 2
                if card.id == Abra:
                    if is_early: score += W["play_abra_early"]
                    elif abra_line_on_field < 3: score += W["play_abra_need"]
                    elif bench_free <= 1: score = -1
                    else: score += W["play_abra_extra"]
                elif card.id in DUNSPARCE_IDS:
                    if dunsparce_line_on_field < 1: score += W["play_dun_first_early"] if is_early else W["play_dun_first_late"]
                    elif dunsparce_line_on_field < 2: score += W["play_dun_second"]
                    elif op_has_ex and len(my_field) < 5: score += W["play_dun_ex"]
                    else: score = -1
                elif card.id == Fezandipiti_ex:
                    score = W["play_fez"] if need_fez else -1
                elif card.id == Genesect:
                    score = W["play_genesect"] if (not op_used_ace_spec and (hand_counts[Lucky_Helmet]
                                     or hand_counts[Handheld_Fan] or hand_counts[Hero_Cape])) else -1
                elif card.id == Psyduck:
                    score = W["play_psyduck"] if op_has_duskull else -1
                elif card.id == Shaymin:
                    score = W["play_shaymin"] if (op_has_water_threat or op_has_grimmsnarl) else -1
                elif card.id == Fan_Rotom:
                    score = W["play_fanrotom"] if state.turn <= 2 else -1
                else:
                    score = -1
                if bench_free <= 1 and score > 0 and card.id != Abra and not op_has_ex:
                    score -= W["play_bench_penalty"]

            else:
                score = 10000
                cid = card.id
                if cid == Buddy_Buddy_Poffin:
                    if safe_draws < 2: score = -1
                    elif state.turn <= 2:
                        score = W["poffin_early"] if (abra_line_on_field < 3 or dunsparce_line_on_field < 1) else W["poffin_fallback"]
                    else:
                        score = W["poffin_late"] if (abra_line_on_field < 3 or dunsparce_line_on_field < 2) else (W["poffin_fallback"] if target_can_kill else -1)
                elif cid == Poke_Pad:
                    if safe_draws < 1 or overdraw: score = -1
                    elif state.turn <= 2: score = W["pokepad_early"]
                    else: score = W["pokepad_need"] if abra_line_on_field < 3 else W["pokepad_ok"]
                elif cid == Switch:
                    score = W["boss_kill"] if (attack_locked and bench_ready_alak) else -1
                elif cid == Rare_Candy:
                    score = W["rare_candy"] if (field_counts[Abra] >= 1 and hand_counts[Alakazam] >= 1 and safe_draws >= 3) else -1
                elif cid == Night_Stretcher:
                    dis_abra = sum(discard_counts[x] for x in ABRA_LINE)
                    if dis_abra >= 1: score = W["night_stretcher_mon"]
                    elif discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy] >= 1: score = W["night_stretcher_energy"]
                    else: score = -1
                elif cid == Sacred_Ash:
                    dis_abra = sum(discard_counts[x] for x in ABRA_LINE)
                    score = W["sacred_ash_hi"] if dis_abra >= 2 else (W["sacred_ash_lo"] if dis_abra >= 1 else -1)
                elif cid == Enhanced_Hammer:
                    if target_hammer_needed > 0: score = W["hammer_target"]
                    elif any(count_special_defense_energies(p) > 0 for p in op_all_pokemon): score = W["hammer_any"]
                    else: score = -1
                elif cid == Wondrous_Patch:
                    score = W["wondrous_patch"] if (discard_counts[Basic_Psychic_Energy] >= 1 and bench_abra_no_energy) else -1
                elif cid == Meddling_Memo:
                    score = W["meddling_memo"] if op_state.handCount >= 5 else -1
                elif cid == Boss_Orders:
                    if len(my_field) <= 1: score = -1
                    elif target_use_boss and target_can_kill: score = W["boss_kill"]
                    else: score = -1
                elif cid == Hilda:
                    score = W["hilda"] if (safe_draws >= 2 and not overdraw) else -1
                elif cid == Dawn:
                    if overdraw: score = -1
                    elif len(my_field) <= 1 and safe_draws >= 3: score = W["dawn_emergency"]
                    elif safe_draws >= 3: score = W["dawn"]
                    else: score = -1
                elif cid == Lillie_Det:
                    if safe_draws < 6: score = -1
                    elif hand_size <= (5 if my_prize_count == 6 else 4): score = W["lillie"]
                    else: score = -1
                elif cid == Lanas_Aid:
                    rec = sum(discard_counts[x] for x in (Abra, Kadabra, Alakazam, Dunsparce65, Dunsparce305, Basic_Psychic_Energy))
                    score = W["lana"] if rec >= 2 else -1
                elif cid == Xerosic:
                    score = W["xerosic"] if op_state.handCount >= 6 else -1
                elif cid == Eri:
                    score = W["eri"] if op_state.handCount >= 4 else -1
                elif cid == Neutralization_Zone:
                    if our_stadium_up and stadium_id == Neutralization_Zone: score = -1
                    elif op_has_ex: score = W["nz_ex"]
                    elif stadium_id != 0 and not our_stadium_up: score = W["nz_counter"]
                    else: score = -1
                elif cid == Battle_Cage:
                    if stadium_id == Battle_Cage: score = -1
                    elif stadium_id != 0 and not our_stadium_up: score = W["cage_counter"]
                    elif op_has_dragapult_line or op_has_grimmsnarl: score = W["cage_snipe"]
                    else: score = -1
                elif cid == Nighttime_Mine:
                    if stadium_id == Nighttime_Mine: score = -1
                    elif stadium_id != 0 and not our_stadium_up: score = W["mine_counter"]
                    else: score = -1
                elif cid == Jamming_Tower:
                    if stadium_id == Jamming_Tower: score = -1
                    elif op_has_tools: score = W["jamming_tools"]
                    elif stadium_id != 0 and not our_stadium_up: score = W["jamming_counter"]
                    else: score = -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)

            if card.id == Lucky_Helmet:
                score = W["helmet"]
                if pokemon.id == Genesect: score += 300
                elif o.inPlayArea == AreaType.ACTIVE: score += 200
                else: score += 50
            elif card.id == Handheld_Fan:
                if pokemon.id in ABRA_LINE:
                    score = W["fan_abra"] + (300 if o.inPlayArea == AreaType.ACTIVE else 0)
                elif pokemon.id == Genesect: score = W["fan_genesect"]
                else: score = -1
            elif card.id == Air_Balloon:
                rc = card_table[pokemon.id].retreatCost
                score = W["balloon"] if (rc >= 1 and o.inPlayArea == AreaType.ACTIVE and pokemon.id not in ABRA_LINE) else -1
            elif card.id == Hero_Cape:
                if pokemon.id == Alakazam:
                    score = W["cape_alak"] + (300 if o.inPlayArea == AreaType.ACTIVE else 0)
                elif pokemon.id == Kadabra and hand_counts[Alakazam] >= 1: score = W["cape_kadabra"]
                elif pokemon.id in ABRA_LINE: score = W["cape_abra"]
                else: score = -1
            elif card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE: score = W["energy_retreat"]
                elif len(pokemon.energyCards) >= 1: score = -1
                elif pokemon.id in ABRA_LINE:
                    score = W["energy_abra"] + {Alakazam: 30, Kadabra: 20, Abra: 10}.get(pokemon.id, 0)
                    if o.inPlayArea == AreaType.ACTIVE: score += 5
                else: score = -1
                if card.id == Telepath_Psychic_Energy and safe_draws < 2 and score > 0: score = -1
            elif card.id == Enriching_Energy:
                if pokemon.id in ABRA_LINE and len(pokemon.energyCards) >= 1:
                    score = W["enriching_2nd"] + (200 if o.inPlayArea == AreaType.ACTIVE else 0)
                elif need_retreat_energy and o.inPlayArea == AreaType.ACTIVE: score = W["energy_retreat"]
                elif pokemon.id in ABRA_LINE: score = W["enriching_1st"]
                else: score = -1
                if safe_draws < 4 and score > 0: score = -1
            elif card.id == Mist_Energy:
                if pokemon.id in ABRA_LINE and len(pokemon.energyCards) >= 1:
                    score = W["mist_2nd"] + (200 if o.inPlayArea == AreaType.ACTIVE else 0)
                elif need_retreat_energy and o.inPlayArea == AreaType.ACTIVE: score = W["mist_retreat"]
                else: score = -1
            else:
                score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = W["evolve_base"]
            if card.id == Alakazam:
                if safe_draws < 3: score = -1
                elif o.inPlayArea == AreaType.ACTIVE: score += 200
                else: score += 50
                score += len(pokemon.energies) * 10
            elif card.id == Kadabra:
                if safe_draws < 2: score = -1
                else:
                    score += 100
                    if len(pokemon.energies) == 0: score += 50
                    elif hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0: score -= 120
            elif card.id == Dudunsparce:
                score = -1 if safe_draws < 2 else score + 80
            else:
                score += 30

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is None: scores.append(0); continue
            if card.id == Dudunsparce:
                if len(my_field) <= 1: score = -1
                elif safe_draws < 3 or overdraw: score = -1
                elif need_dudunsparce_draw: score = W["ability_dudun"]
                elif hand_size < (9 if op_has_ex else 6): score = W["ability_dudun"]
                elif active_id not in ABRA_LINE and o.area == AreaType.ACTIVE: score = W["ability_dudun"]
                else: score = -1
            elif card.id == Fezandipiti_ex:
                score = W["ability_fez"] if (need_fez and safe_draws >= 3) else -1
            elif card.id == Fan_Rotom:
                score = W["ability_fanrotom"] if state.turn <= 2 else -1
            elif card.id in OUR_STADIUMS:
                score = 1
            else:
                score = W["ability_default"]

        elif o.type == OptionType.RETREAT:
            if attack_locked and bench_ready_alak: score = W["boss_kill"]
            elif active_id == Alakazam and active_has_psychic: score = -1
            elif use_kadabra_finish and active_id != Kadabra and field_counts[Kadabra] >= 1: score = W["retreat_kadabra"]
            elif active_id in (Abra, Dunsparce65, Dunsparce305, Dudunsparce, Psyduck, Shaymin, Genesect, Fan_Rotom):
                score = W["retreat_promote"] if (field_counts[Alakazam] >= 1 or field_counts[Kadabra] >= 1) else -1
            else: score = -1

        elif o.type == OptionType.ATTACK:
            score = W["attack_base"]
            if o.attackId == ATTACK_POWERFUL_HAND: score += W["attack_powerful"]
            elif o.attackId == ATTACK_SUPER_PSY_BOLT: score += W["attack_psybolt_kill"] if op_active_hp <= 30 else W["attack_psybolt"]
            elif o.attackId == ATTACK_TELEPORTATION: score += W["attack_teleport"]

        scores.append(score)
    return scores


def _post_pick(obs, picked_idx):
    global ability_used_dudunsparce, ability_used_fezandipiti
    sel = obs.select
    if sel.context != SelectContext.MAIN: return
    o = sel.option[picked_idx]
    if o.type == OptionType.ABILITY:
        card = get_card(obs, o.area, o.index, obs.current.yourIndex)
        if card is not None:
            if card.id == Dudunsparce: ability_used_dudunsparce = True
            elif card.id == Fezandipiti_ex: ability_used_fezandipiti = True


def _agent_impl(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None: return my_deck
    scores = heuristic_scores(obs)
    desc = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if desc: _post_pick(obs, desc[0])
    return desc[:obs.select.maxCount]


def _heuristic_agent(obs_dict):
    try:
        return _agent_impl(obs_dict)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None: return my_deck
            n = len(obs.select.option)
            k = min(max(1, obs.select.minCount), n) if n else 0
            return list(range(k))
        except Exception:
            return [0]

# ==================== SEARCH LAYER (inlined) ====================
import time, random
from collections import Counter
try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False
USE_SEARCH = True          # local_battle.py flips this off unless KEEP_SEARCH=1
N_DET = 3                  # determinizations
K_OPP = 3                  # opponent branching at ply-2 MAIN
MAX_SUBSTEPS = 40          # per greedy-complete rollout
TIME_BUDGET_S = 0.80
SEARCH_MAX_OPTS = 24
DUMMY_BASIC = 646          # Impidimp (any basic)
DUMMY_ENERGY = 7           # Dark

_search_ok = _SEARCH_IMPORT_OK
_search_reported = False
_stats = {"calls": 0, "ms": 0.0, "overrides": 0, "considered": 0, "fail": 0}

# ---- archetype templates for opponent belief ----
_TEMPLATES = []  # list[(name, Counter, list[int])]
for _d in ("/kaggle_simulations/agent/top20_decks", "top20_decks", "../top20_decks", os.path.join(os.path.dirname(os.path.abspath(".")), "top20_decks")):
    if os.path.isdir(_d):
        for _fn in sorted(os.listdir(_d)):
            if not _fn.endswith(".csv"):
                continue
            try:
                with open(os.path.join(_d, _fn)) as _f:
                    _ids = [int(x) for x in _f.read().split() if x.strip()][:60]
                if len(_ids) == 60:
                    _TEMPLATES.append((_fn, Counter(_ids), _ids))
            except Exception:
                pass
        break

# EnergyType â†’ basic energy card ID (card IDs 1-8 mirror EnergyType 1-8)
_BASIC_ENERGY = {i: i for i in range(1, 9)}
# per-template signature = pokemon card IDs in it (for overlap match)
_card_data = {c.cardId: c for c in all_card_data()}
def _pokemon_ids(counter):
    return {cid for cid in counter if _card_data.get(cid) and
            _card_data[cid].cardType == CardType.POKEMON}
_TEMPLATE_SIG = [(n, _pokemon_ids(c), c, ids) for (n, c, ids) in _TEMPLATES]


# ==================== belief model ====================

def _my_visible(state, me_i):
    """Multiset of our own card IDs whose location is known (not deck/facedown-prize)."""
    me = state.players[me_i]
    seen = Counter()
    for c in me.hand or []:
        seen[c.id] += 1
    for c in me.discard:
        seen[c.id] += 1
    for c in me.prize:
        if c is not None:
            seen[c.id] += 1
    for p in me.active + me.bench:
        if p is None:
            continue
        seen[p.id] += 1
        for c in p.energyCards: seen[c.id] += 1
        for c in p.tools: seen[c.id] += 1
        for c in p.preEvolution: seen[c.id] += 1
    if state.stadium and state.stadium[0].playerIndex == me_i:
        seen[state.stadium[0].id] += 1
    return seen


def _op_visible(state, op_i):
    """Opponent card IDs we can observe (field + discard + stadium). Also returns
    the dominant EnergyType seen on their field for fallback synthesis."""
    op = state.players[op_i]
    seen = Counter()
    etype = Counter()
    for c in op.discard:
        seen[c.id] += 1
    for p in op.active + op.bench:
        if p is None:
            continue
        seen[p.id] += 1
        for c in p.energyCards: seen[c.id] += 1
        for c in p.tools: seen[c.id] += 1
        for c in p.preEvolution: seen[c.id] += 1
        for e in p.energies:
            etype[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_i:
        seen[state.stadium[0].id] += 1
    # face-up prizes (rare)
    for c in op.prize:
        if c is not None:
            seen[c.id] += 1
    return seen, etype


def _match_archetype(op_seen):
    """Pick template with highest pokemon-ID overlap vs opponent's visible cards."""
    op_mons = {cid for cid in op_seen if _card_data.get(cid) and
               _card_data[cid].cardType == CardType.POKEMON}
    if not op_mons or not _TEMPLATE_SIG:
        return None
    best, best_n = None, 0
    for name, sig, cnt, ids in _TEMPLATE_SIG:
        n = len(sig & op_mons)
        if n > best_n:
            best, best_n = (cnt, ids), n
    return best if best_n >= 1 else None


def _sample_hidden(state, me_i):
    """One determinization of all hidden zones. Returns kwargs for search_begin."""
    me = state.players[me_i]
    op_i = 1 - me_i
    op = state.players[op_i]

    # --- our deck + facedown prizes ---
    seen = _my_visible(state, me_i)
    remain = []
    for cid, n in Counter(my_deck).items():
        remain.extend([cid] * max(0, n - seen.get(cid, 0)))
    n_prize_hidden = sum(1 for c in me.prize if c is None)
    need = me.deckCount + n_prize_hidden
    if len(remain) < need:
        remain += [DUMMY_ENERGY] * (need - len(remain))
    random.shuffle(remain)
    your_deck = remain[:me.deckCount]
    # search_begin wants full prize array; fill facedown slots from remainder
    fill = iter(remain[me.deckCount:need])
    your_prize = [c.id if c is not None else next(fill, DUMMY_ENERGY) for c in me.prize]

    # --- opponent hidden (deck + prize + hand) ---
    op_seen, etype = _op_visible(state, op_i)
    tpl = _match_archetype(op_seen)
    if tpl is not None:
        cnt, _ = tpl
        pool = []
        for cid, n in cnt.items():
            pool.extend([cid] * max(0, n - op_seen.get(cid, 0)))
    else:
        # fallback: most-seen visible card + basic energy of their dominant type
        etop = max(etype.items(), key=lambda x: x[1])[0] if etype else 7
        energy_id = _BASIC_ENERGY.get(etop, 7)
        top_card = None
        if op_seen:
            top_card = max(op_seen.items(), key=lambda x: x[1])[0]
        pool = ([top_card] * 30 if top_card else []) + [energy_id] * 30
        # ensure at least a few basics so sim doesn't wedge
        pool += [DUMMY_BASIC] * 8

    n_op_prize_hidden = sum(1 for c in op.prize if c is None)
    op_need = op.deckCount + n_op_prize_hidden + op.handCount
    if len(pool) < op_need:
        # pad with template basics or dark energy
        pad = [DUMMY_ENERGY] * (op_need - len(pool))
        pool += pad
    random.shuffle(pool)
    opponent_deck = pool[:op.deckCount]
    off = op.deckCount
    fill_op = iter(pool[off:off + n_op_prize_hidden])
    opponent_prize = [c.id if c is not None else next(fill_op, DUMMY_ENERGY)
                      for c in op.prize]
    off += n_op_prize_hidden
    opponent_hand = pool[off:off + op.handCount]
    opponent_active = [DUMMY_BASIC] if (op.active and op.active[0] is None) else []

    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)


# ==================== leaf eval + rollout helpers ====================

def _leaf_eval(state, me_i):
    if state is None:
        return 0.0
    if state.result is not None and state.result >= 0:
        if state.result == me_i: return 1e7
        if state.result == 2: return 0.0
        return -1e7
    me = state.players[me_i]
    op = state.players[1 - me_i]
    my_field = [p for p in (me.active + me.bench) if p]
    op_field = [p for p in (op.active + op.bench) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(p.energies) for p in my_field)
    op_en = sum(len(p.energies) for p in op_field)
    no_active = 0 if (me.active and me.active[0]) else 1
    return (1000.0 * (len(op.prize) - len(me.prize))
            + my_hp - op_hp
            + 5.0 * (my_en - op_en)
            - 4000.0 * no_active)


def _greedy_pick(obs):
    """Return (choice, order) using grimm1 heuristic. Works for either player â€”
    unknown-card options fall through to generic scores in heuristic_scores."""
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return [], []
    try:
        sc = heuristic_scores(obs)
    except Exception:
        sc = list(range(n, 0, -1))
    order = sorted(range(n), key=lambda i: sc[i], reverse=True)
    k = min(sel.maxCount, n)
    k = max(k, min(max(1, sel.minCount), n))
    return order[:k], order


def _greedy_complete_turn(sid, cur, owner, deadline):
    """Step greedily while it's `owner`'s turn. Stops at hand-off / terminal /
    budget. Returns (sid, obs)."""
    for _ in range(MAX_SUBSTEPS):
        if time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or (cs.result is not None and cs.result >= 0):
            return sid, cur
        if cs.yourIndex != owner or cur.select is None:
            return sid, cur
        choice, _ = _greedy_pick(cur)
        if not choice:
            return sid, cur
        try:
            ss = search_step(sid, choice)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _advance_forced(sid, cur, owner, deadline, limit=8):
    """Resolve non-MAIN sub-selects for `owner` (e.g. promote after KO,
    setup active) so we reach a MAIN decision or hand-off."""
    for _ in range(limit):
        if time.monotonic() > deadline:
            break
        cs = cur.current
        if (cs is None or cur.select is None or cs.yourIndex != owner
                or cur.select.context == SelectContext.MAIN
                or (cs.result is not None and cs.result >= 0)):
            break
        ch, _ = _greedy_pick(cur)
        if not ch:
            break
        try:
            ss = search_step(sid, ch)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


# ==================== 2-ply minimax ====================

def _search_decide(obs, base_order, base_scores):
    global _search_ok, _search_reported
    if not (USE_SEARCH and _search_ok):
        return None
    st = obs.current
    sel = obs.select
    if st is None or sel is None or sel.context != SelectContext.MAIN:
        return None
    n = len(sel.option)
    if n < 3 or n > SEARCH_MAX_OPTS or st.turn < 2:
        return None
    if getattr(obs, "search_begin_input", None) is None:
        if not _search_reported:
            print("[alak_evo2_s] search_begin_input=None â†’ search disabled", file=sys.stderr)
            _search_reported = True
        _search_ok = False
        return None

    me_i = st.yourIndex
    heur_top = base_order[0]
    # candidates: heuristic top-8 (heur_top + non-terminal). Terminal actions
    # (ATTACK/END) reached via greedy rollout so no need to branch on them.
    cand = [heur_top]
    for i in base_order[1:]:
        if sel.option[i].type in (OptionType.ATTACK, OptionType.END):
            continue
        if base_scores[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= 8:
            break
    if len(cand) < 2:
        return None

    t0 = time.monotonic()
    deadline = t0 + TIME_BUDGET_S
    acc = {i: 0.0 for i in cand}
    n_eval = {i: 0 for i in cand}
    began = False
    try:
        for det in range(N_DET):
            if time.monotonic() > deadline:
                break
            hidden = _sample_hidden(st, me_i)
            try:
                ss0 = search_begin(obs, **hidden)
                began = True
            except Exception as e:
                if not _search_reported:
                    print(f"[alak_evo2_s] search_begin failed: {e!r} â†’ disabled", file=sys.stderr)
                    _search_reported = True
                _search_ok = False
                return None
            root_sid = ss0.searchId

            for idx in cand:
                if time.monotonic() > deadline:
                    break
                # ply 1: take idx, greedy-complete our turn
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid1, cur = ss.searchId, ss.observation
                sid1, cur = _greedy_complete_turn(sid1, cur, me_i, deadline)
                cs = cur.current
                if (cs is None or (cs.result is not None and cs.result >= 0)
                        or cs.yourIndex == me_i or cur.select is None):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue
                # advance opponent's forced sub-selects to MAIN
                sid1, cur = _advance_forced(sid1, cur, 1 - me_i, deadline)
                cs = cur.current
                if (cs is None or cur.select is None
                        or cur.select.context != SelectContext.MAIN
                        or cs.yourIndex == me_i):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue
                # ply 2: min over opponent's top-K first-actions
                _, op_order = _greedy_pick(cur)
                worst = None
                for k in range(min(K_OPP, len(op_order))):
                    if time.monotonic() > deadline:
                        break
                    try:
                        ss2 = search_step(sid1, [op_order[k]])
                    except Exception:
                        continue
                    sid2, cur2 = ss2.searchId, ss2.observation
                    sid2, cur2 = _greedy_complete_turn(sid2, cur2, 1 - me_i, deadline)
                    # resolve our forced replacement (promote after KO)
                    sid2, cur2 = _advance_forced(sid2, cur2, me_i, deadline, limit=6)
                    v = _leaf_eval(cur2.current, me_i)
                    worst = v if worst is None else min(worst, v)
                if worst is None:
                    worst = _leaf_eval(cs, me_i)
                acc[idx] += worst
                n_eval[idx] += 1

            try:
                search_end()
            except Exception:
                pass
            began = False

        elapsed = time.monotonic() - t0
        _stats["calls"] += 1
        _stats["ms"] += elapsed * 1000.0
        # pick: avg over dets, tie-break by heuristic score. Only compare
        # candidates evaluated in the same #dets as heur_top (fair sample).
        n_top = n_eval.get(heur_top, 0)
        if n_top == 0:
            return None
        evaluated = [i for i in cand if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] + 1e-6 * base_scores[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        _stats["considered"] += 1
        if best == heur_top:
            return None
        # override only when search shows a real margin (â‰¥ half a prize)
        if avg[best] < avg[heur_top] + 500.0:
            return None
        _stats["overrides"] += 1
        return best
    except Exception as e:
        _stats["fail"] += 1
        if not _search_reported:
            print(f"[alak_evo2_s] search crashed: {e!r} â†’ fallback", file=sys.stderr)
            _search_reported = True
        return None
    finally:
        if began:
            try:
                search_end()
            except Exception:
                pass


# ==================== entry ====================

def agent(obs_dict):
    global _search_ok
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return _heuristic_agent(obs_dict)
    if obs.select is None:
        _search_ok = _SEARCH_IMPORT_OK  # new game
        return my_deck

    # base heuristic (incl. ML residual) as fallback + candidate ordering
    fallback = _heuristic_agent(obs_dict)
    sel = obs.select
    if sel.context != SelectContext.MAIN:
        return fallback
    try:
        base_scores = heuristic_scores(obs)
        n = len(sel.option)
        base_order = sorted(range(n), key=lambda i: base_scores[i], reverse=True)
    except Exception:
        return fallback

    pick = _search_decide(obs, base_order, base_scores)
    if pick is None:
        return fallback
    return [pick]


\

# v19: static templates are public reference data.  They are activated only
# after the matching opponent line is visible in the observation.
_GRIMM_IDS = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259]
_TUSK_IDS = [58, 58, 58, 58, 344, 344, 344, 344, 345, 345, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1086, 1122, 1122, 1122, 1122, 1121, 1123, 1123, 1123, 1123, 1197, 1197, 1197, 1197, 1185, 1185, 1185, 1185, 1182, 1182, 1182, 1182, 1204, 1204, 1194, 1194, 1247, 1147, 20, 20, 20, 20, 11, 11, 11, 11, 345, 345, 607]
_GRIMM_TEMPLATE_SIG = [
    ("grimmsnarl_public", _pokemon_ids(Counter(_GRIMM_IDS)), Counter(_GRIMM_IDS), _GRIMM_IDS)
]
_TUSK_TEMPLATE_SIG = [
    ("great_tusk_public", _pokemon_ids(Counter(_TUSK_IDS)), Counter(_TUSK_IDS), _TUSK_IDS)
]
_GRIMM_LINE = {646, 647, 648}
_TUSK_LINE = {58, 344, 345, 607}
TEAM_ROCKET_ENERGY = 15
_base_heuristic_scores = heuristic_scores
_base_agent = agent


def _visible_opponent_line(obs_dict, card_ids):
    try:
        obs = to_observation_class(obs_dict)
        state = obs.current
        if state is None:
            return False
        opponent = state.players[1 - state.yourIndex]
        cards = list(opponent.active or []) + list(opponent.bench or []) + list(opponent.discard or [])
        return any(card is not None and card.id in card_ids for card in cards)
    except Exception:
        return False


def _rocket_energy_hammer_scores(obs):
    scores = _base_heuristic_scores(obs)
    state, select = obs.current, obs.select
    if state is None or select is None or select.context != SelectContext.MAIN:
        return scores
    opponent = state.players[1 - state.yourIndex]
    if not any(
        energy.id == TEAM_ROCKET_ENERGY
        for pokemon in opponent.active + opponent.bench
        if pokemon is not None
        for energy in pokemon.energyCards
    ):
        return scores
    for index, option in enumerate(select.option):
        if option.type != OptionType.PLAY:
            continue
        card = get_card(obs, AreaType.HAND, option.index, state.yourIndex)
        if card is not None and card.id == Enhanced_Hammer:
            scores[index] += 20_000
    return scores


heuristic_scores = _rocket_energy_hammer_scores


def agent(obs_dict, configuration=None):
    global _TEMPLATE_SIG
    templates = []
    if _visible_opponent_line(obs_dict, _GRIMM_LINE):
        templates.extend(_GRIMM_TEMPLATE_SIG)
    if _visible_opponent_line(obs_dict, _TUSK_LINE):
        templates.extend(_TUSK_TEMPLATE_SIG)
    _TEMPLATE_SIG = templates
    return _base_agent(obs_dict)


# Standalone submission boundary: resolve the exact competition deck without
# assuming that Kaggle executes main.py from the archive directory.
_standalone_policy_agent = agent


def _standalone_competition_deck():
    candidates = []
    source_file = globals().get("__file__")
    if source_file:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(source_file)), "deck.csv"))
    candidates.extend([
        os.path.join(os.getcwd(), "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ])
    for candidate in candidates:
        try:
            with open(candidate, "r") as handle:
                cards = [int(value) for value in handle.read().splitlines() if value.strip()]
            if len(cards) == 60:
                return cards
        except Exception:
            pass
    return list(my_deck)


def agent(obs_dict, configuration=None):
    global my_deck
    if (
        isinstance(obs_dict, dict)
        and obs_dict.get("current") is None
        and obs_dict.get("select") is None
    ):
        my_deck = _standalone_competition_deck()
        return list(my_deck)
    return _standalone_policy_agent(obs_dict)


def competition_entrypoint(obs_dict):
    return agent(obs_dict)
