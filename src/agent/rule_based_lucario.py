import os
import sys
from collections import defaultdict

# Add project root to sys.path if running outside kaggle
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AreaType:
    DECK = 0
    HAND = 1
    DISCARD = 2
    ACTIVE = 3
    BENCH = 4
    LOST_ZONE = 5

class CardType:
    POKEMON = 1
    TRAINER = 2
    ENERGY = 3

class EnergyType:
    FIGHTING = 1

class SelectContext:
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    LOOK = 1
    TO_HAND = 2
    ATTACH_FROM = 3
    ATTACH_TO = 4
    EVOLVE_FROM = 5
    DISCARD = 5
    SWITCH_ACTIVE_POKEMON = 7

class OptionType:
    MAIN = 0
    SELECT_CARD = 3
    NUMBER = 2
    YES = 9
    CARD = 1
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    RETREAT = 12
    ATTACK = 13
    END_TURN = 14

import json
from types import SimpleNamespace

def to_observation_class(d):
    return json.loads(json.dumps(d), object_hook=lambda x: SimpleNamespace(**x))


import json

WEIGHTS = {
    "play_pokemon_base": 20000,
    "play_dusk_pad": 8000,
    "play_switch": 6000,
    "play_premium": 5000,
    "play_boss": 3200,
    "play_carmine": 3000,
    "play_lillie": 3100,
    "attach_hero_cape": 7000,
    "evolve_base": 9000,
    "ability_base": 30000,
    "retreat_base": 2000,
    "attack_base": 1000
}

# Try loading runtime overrides
for _p in ("lucario_w.json", "./lucario_w.json", "/kaggle_simulations/agent/lucario_w.json"):
    if os.path.exists(_p):
        try:
            WEIGHTS.update(json.load(open(_p)))
        except Exception:
            pass
        break

W = WEIGHTS

try:
    all_card = all_card_data()
    card_table = {c.cardId:c for c in all_card}
except:
    card_table = {}

def safe_card_data(card_id):
    if card_id in card_table:
        return card_table[card_id]
    return SimpleNamespace(megaEx=False, ex=False, weakness=None, stage1=False, stage2=False)

# Decklist
Makuhita = 673  # ×2
Hariyama = 674  # ×2
Lunatone = 675  # ×2
Solrock = 676  # ×3
Riolu = 677  # ×3
Mega_Lucario_ex = 678  # ×4
Dusk_Ball = 1102  # ×4
Switch = 1123  # ×2
Premium_Power_Pro = 1141  # ×4
Fighting_Gong = 1142  # ×4
Poke_Pad = 1152  # x4
Hero_Cape = 1159  # ×1
Boss_Orders = 1182  # ×2
Carmine = 1192  # ×4
Lillie_Determination = 1227  # ×4
Gravity_Mountain = 1252  # ×2
Basic_Fighting_Energy = 6  # ×13

class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False

plan = AttackPlan()
pre_turn = 0
ability_used = False

def get_card(obs, area, index: int, player_index: int):
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        elif area == AreaType.HAND:
            return ps.hand[index]
        elif area == AreaType.DISCARD:
            return ps.discard[index]
        elif area == AreaType.ACTIVE:
            return ps.active[index]
        elif area == AreaType.BENCH:
            return ps.bench[index]
        elif area == AreaType.PRIZE:
            return ps.prize[index]
        elif area == AreaType.STADIUM:
            return obs.current.stadium[index]
        elif area == AreaType.LOOKING:
            return obs.current.looking[index]
    except (IndexError, AttributeError):
        pass
    return None

def prize_count(pokemon) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = safe_card_data(pokemon.id)
    count = 3 if getattr(data, 'megaEx', False) else 2 if getattr(data, 'ex', False) else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
            count -= 1
    return max(0, count)

def pokemon_score(pokemon) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = safe_card_data(pokemon.id)
    score = prize_count(pokemon) * 1000
    score += len(getattr(pokemon, 'energies', [])) * 150
    score += len(getattr(pokemon, 'tools', [])) * 100
    if getattr(data, 'stage2', False):
        score += 250
    elif getattr(data, 'stage1', False):
        score += 130
    
    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


# ==================== SEARCH LAYER ====================
import time, random
from collections import Counter
try:
    from cg.api import search_begin, search_step, search_end  # type: ignore
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False

USE_SEARCH = True
N_DET = 2
K_OPP = 2
MAX_SUBSTEPS = 40
TIME_BUDGET_S = 0.80
SEARCH_MAX_OPTS = 24
DUMMY_BASIC = 677 # Riolu
DUMMY_ENERGY = 6  # Fighting

_search_ok = _SEARCH_IMPORT_OK
_search_reported = False

def _my_visible(state, me_i):
    me = state.players[me_i]
    seen = Counter()
    for c in me.hand or []: seen[c.id] += 1
    for c in me.discard: seen[c.id] += 1
    for c in me.prize:
        if c is not None: seen[c.id] += 1
    for p in me.active + me.bench:
        if p is None: continue
        seen[p.id] += 1
        for c in p.energyCards: seen[c.id] += 1
        for c in p.tools: seen[c.id] += 1
        for c in getattr(p, 'preEvolution', []): seen[c.id] += 1
    if state.stadium and state.stadium[0].playerIndex == me_i:
        seen[state.stadium[0].id] += 1
    return seen

def _op_visible(state, op_i):
    op = state.players[op_i]
    seen = Counter()
    etype = Counter()
    for c in op.discard: seen[c.id] += 1
    for p in op.active + op.bench:
        if p is None: continue
        seen[p.id] += 1
        for c in p.energyCards: seen[c.id] += 1
        for c in p.tools: seen[c.id] += 1
        for c in getattr(p, 'preEvolution', []): seen[c.id] += 1
        for e in getattr(p, 'energies', []): etype[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_i:
        seen[state.stadium[0].id] += 1
    for c in op.prize:
        if c is not None: seen[c.id] += 1
    return seen, etype

def _sample_hidden(state, me_i, my_deck):
    me = state.players[me_i]
    op_i = 1 - me_i
    op = state.players[op_i]
    
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
    fill = iter(remain[me.deckCount:need])
    your_prize = [c.id if c is not None else next(fill, DUMMY_ENERGY) for c in me.prize]

    op_seen, etype = _op_visible(state, op_i)
    etop = max(etype.items(), key=lambda x: x[1])[0] if etype else 1
    top_card = max(op_seen.items(), key=lambda x: x[1])[0] if op_seen else None
    
    pool = ([top_card] * 30 if top_card else []) + [etop] * 30 + [DUMMY_BASIC] * 8
    n_op_prize_hidden = sum(1 for c in op.prize if c is None)
    op_need = op.deckCount + n_op_prize_hidden + op.handCount
    if len(pool) < op_need:
        pool += [DUMMY_ENERGY] * (op_need - len(pool))
    random.shuffle(pool)
    opponent_deck = pool[:op.deckCount]
    off = op.deckCount
    fill_op = iter(pool[off:off + n_op_prize_hidden])
    opponent_prize = [c.id if c is not None else next(fill_op, DUMMY_ENERGY) for c in op.prize]
    off += n_op_prize_hidden
    opponent_hand = pool[off:off + op.handCount]
    opponent_active = [DUMMY_BASIC] if (op.active and op.active[0] is None) else []
    
    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)

def _leaf_eval(state, me_i):
    if state is None: return 0.0
    if getattr(state, 'result', None) is not None and state.result >= 0:
        if state.result == me_i: return 1e7
        if state.result == 2: return 0.0
        return -1e7
    me = state.players[me_i]
    op = state.players[1 - me_i]
    my_field = [p for p in (me.active + me.bench) if p]
    op_field = [p for p in (op.active + op.bench) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(getattr(p, 'energies', [])) for p in my_field)
    op_en = sum(len(getattr(p, 'energies', [])) for p in op_field)
    no_active = 0 if (me.active and me.active[0]) else 1
    
    # Custom Lucario bonuses
    lucario_bonus = sum(500 for p in my_field if getattr(p, 'id', -1) == 678)
    
    return (1000.0 * (len(op.prize) - len(me.prize))
            + my_hp - op_hp
            + 5.0 * (my_en - op_en)
            - 4000.0 * no_active
            + lucario_bonus)

def _greedy_complete_turn(sid, cur, owner, deadline, agent_func):
    for _ in range(MAX_SUBSTEPS):
        if time.monotonic() > deadline: break
        cs = getattr(cur, 'current', None)
        if cs is None or (getattr(cs, 'result', None) is not None and cs.result >= 0): break
        if cs.yourIndex != owner or getattr(cur, 'select', None) is None: break
        
        # Call agent_func for choice
        try:
            ch_list = agent_func(cur.__dict__ if hasattr(cur, '__dict__') else cur)
            choice = ch_list[:1] if ch_list else []
        except Exception:
            choice = []
        if not choice: break
        try:
            ss = search_step(sid, choice)
            sid, cur = ss.searchId, ss.observation
        except Exception:
            break
    return sid, cur

def _advance_forced(sid, cur, owner, deadline, agent_func, limit=8):
    for _ in range(limit):
        if time.monotonic() > deadline: break
        cs = getattr(cur, 'current', None)
        if (cs is None or getattr(cur, 'select', None) is None or cs.yourIndex != owner
            or cur.select.context == 0 or (getattr(cs, 'result', None) is not None and cs.result >= 0)):
            break
        try:
            ch_list = agent_func(cur.__dict__ if hasattr(cur, '__dict__') else cur)
            ch = ch_list[:1] if ch_list else []
        except Exception:
            ch = []
        if not ch: break
        try:
            ss = search_step(sid, ch)
            sid, cur = ss.searchId, ss.observation
        except Exception:
            break
    return sid, cur

def _search_decide(obs, base_order, base_scores, my_deck, agent_func):
    global _search_ok, _search_reported
    if not (USE_SEARCH and _search_ok): return None
    st = getattr(obs, 'current', None)
    sel = getattr(obs, 'select', None)
    if st is None or sel is None or sel.context != 0: return None
    n = len(sel.option)
    if n < 3 or n > SEARCH_MAX_OPTS or st.turn < 2: return None
    if getattr(obs, "search_begin_input", None) is None:
        if not _search_reported:
            _search_reported = True
        _search_ok = False
        return None

    me_i = st.yourIndex
    if not base_order: return None
    heur_top = base_order[0]
    cand = [heur_top]
    for i in base_order[1:]:
        if getattr(sel.option[i], 'type', -1) in (13, 14): # ATTACK, END_TURN
            continue
        if base_scores[i] < 0: continue
        cand.append(i)
        if len(cand) >= 4: break
    if len(cand) < 2: return None

    t0 = time.monotonic()
    deadline = t0 + TIME_BUDGET_S
    acc = {i: 0.0 for i in cand}
    n_eval = {i: 0 for i in cand}
    
    try:
        for det in range(N_DET):
            if time.monotonic() > deadline: break
            hidden = _sample_hidden(st, me_i, my_deck)
            try:
                ss0 = search_begin(obs, **hidden)
            except Exception as e:
                _search_ok = False
                return None
            root_sid = ss0.searchId

            for idx in cand:
                if time.monotonic() > deadline: break
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid1, cur = ss.searchId, ss.observation
                sid1, cur = _greedy_complete_turn(sid1, cur, me_i, deadline, agent_func)
                cs = getattr(cur, 'current', None)
                if (cs is None or (getattr(cs, 'result', None) is not None and cs.result >= 0)
                        or cs.yourIndex == me_i or getattr(cur, 'select', None) is None):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue
                
                sid1, cur = _advance_forced(sid1, cur, 1 - me_i, deadline, agent_func)
                cs = getattr(cur, 'current', None)
                if (cs is None or getattr(cur, 'select', None) is None
                        or cur.select.context != 0 or cs.yourIndex == me_i):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue
                
                # OP turn
                try:
                    op_ch = agent_func(cur.__dict__ if hasattr(cur, '__dict__') else cur)
                except Exception:
                    op_ch = []
                worst = None
                for k in range(min(K_OPP, len(op_ch))):
                    if time.monotonic() > deadline: break
                    try:
                        ss2 = search_step(sid1, [op_ch[k]])
                    except Exception:
                        continue
                    sid2, cur2 = ss2.searchId, ss2.observation
                    sid2, cur2 = _greedy_complete_turn(sid2, cur2, 1 - me_i, deadline, agent_func)
                    sid2, cur2 = _advance_forced(sid2, cur2, me_i, deadline, agent_func, limit=6)
                    v = _leaf_eval(getattr(cur2, 'current', None), me_i)
                    worst = v if worst is None else min(worst, v)
                
                if worst is None: worst = _leaf_eval(cs, me_i)
                acc[idx] += worst
                n_eval[idx] += 1
            try: search_end()
            except Exception: pass
            
        n_top = n_eval.get(heur_top, 0)
        if n_top == 0: return None
        evaluated = [i for i in cand if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] + 1e-6 * base_scores[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        
        if best == heur_top: return None
        if avg[best] < avg[heur_top] + 500.0: return None
        return best
    except Exception:
        return None
# ======================================================

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function."""
    try:
        obs = to_observation_class(obs_dict)
        if obs.select == None:
            # Load deck.csv in the dataset dynamically
            file_path = "deck.csv"
            if not os.path.exists(file_path):
                file_path = "/kaggle_simulations/agent/" + file_path
            with open(file_path, "r") as file:
                csv = file.read().split("\n")
            my_deck = []
            for i in range(60):
                my_deck.append(int(csv[i]))
            return my_deck
            
        state = obs.current
        select = obs.select
        context = select.context
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]
        my_prize = len(my_state.prize)

        global plan
        global pre_turn
        global ability_used
        if pre_turn != state.turn:
            pre_turn = state.turn
            plan = AttackPlan()
            ability_used = False
                
        field_counts = defaultdict(int)
        hand_counts = defaultdict(int)
        discard_counts = defaultdict(int)

        attacker1 = False
        attacker2 = False
        for card in my_state.active + my_state.bench:
            if card == None:
                continue
            field_counts[card.id] += 1
            if card.id == Makuhita or card.id == Hariyama:
                if len(card.energies) >= 3:
                    attacker2 = True
            elif card.id == Riolu or card.id == Mega_Lucario_ex:
                if len(card.energies) >= 2:
                    attacker1 = True

        for card in my_state.hand:
            hand_counts[card.id] += 1

        for card in my_state.discard:
            discard_counts[card.id] += 1

        stadium_id = 0
        for card in state.stadium:
            stadium_id = card.id
                
        can_attack = False
        if context == SelectContext.MAIN:
            can_switch = False
            can_op_switch = False
            can_use_mega_brave = False
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, getattr(o, 'index', 0), my_index)
                    if card is not None:
                        if card.id == Switch:
                            can_switch = True
                        elif card.id == Boss_Orders:
                            can_op_switch = True
                elif o.type == OptionType.EVOLVE:
                    card = get_card(obs, AreaType.HAND, getattr(o, 'index', 0), my_index)
                    if card is not None:
                        if card.id == Hariyama:
                            can_op_switch = True
                elif o.type == OptionType.RETREAT:
                    can_switch = True
                elif o.type == OptionType.ATTACK:
                    can_attack = True
                    if o.attackId == 983:  # Mega Brave
                        can_use_mega_brave = True
            
            my_cards = [my_state.active[0]]
            for pokemon in my_state.bench:
                my_cards.append(pokemon)
            op_cards = [op_state.active[0]]
            for pokemon in op_state.bench:
                op_cards.append(pokemon)

            if state.turn >= 2:
                best_score = -1
                for i, my_pokemon in enumerate(my_cards):
                    if i != 0 and not can_switch:
                        break
                    for a in range(2):
                        energy_required = 0
                        base_damage = 0
                        base_score = 0
                        if my_pokemon.id == Mega_Lucario_ex:
                            if a == 0:
                                energy_required = 1
                                base_damage = 130
                                base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                            else:
                                energy_required = 2
                                base_damage = 270
                            if my_prize == 2 or my_prize == 3:
                                base_score -= 500
                        elif a == 1:
                            break
                        elif my_pokemon.id == Hariyama:
                            energy_required = 3
                            base_damage = 210
                        elif my_pokemon.id == Makuhita:
                            for o in select.option:
                                if o.type == OptionType.EVOLVE:
                                    index = o.inPlayIndex
                                    if o.inPlayArea == AreaType.BENCH:
                                        index += 1
                                    if index == i:
                                        break
                            else:
                                break
                            base_score -= 100
                            energy_required = 3
                            base_damage = 210
                        elif my_pokemon.id == Solrock:
                            if field_counts[Lunatone] >= 1:
                                energy_required = 1
                                base_damage = 70
                        
                        if base_damage <= 0:
                            continue
                        
                        more_energy = False
                        energy_count = len(my_pokemon.energies)
                        if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                            break
                        if energy_count < energy_required:
                            if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                                energy_count += 1
                                if energy_count < energy_required:
                                    continue
                                else:
                                    more_energy = True
                            else:
                                continue

                        for j, op_pokemon in enumerate(op_cards):
                            if j != 0 and not can_op_switch:
                                break
                            damage = base_damage
                            data = safe_card_data(op_pokemon.id)
                            if getattr(data, 'weakness', None) == EnergyType.FIGHTING:
                                damage *= 2
                            elif getattr(data, 'resistance', None) == EnergyType.FIGHTING:
                                damage -= 30
                            prize = 0
                            score = pokemon_score(op_pokemon)
                            if op_pokemon.hp <= damage:
                                prize = prize_count(op_pokemon)
                            else:
                                score *= damage / op_pokemon.hp
                            score += base_score
                                
                            if len(op_state.prize) <= prize:
                                score = 50000
                            
                            if i == 0:
                                score += 220
                            if j == 0:
                                score += 300
                            score += energy_count
                            if best_score < score:
                                best_score = score
                                plan.attacker = i
                                plan.target = j
                                plan.attack_index = a
                                plan.remain_hp = op_pokemon.hp - damage
                                plan.energy = more_energy
        
        def energy_score(pokemon, active: bool) -> int:
            energy_count = len(getattr(pokemon, 'energies', []))
            score = 8000
            if active:
                score += 10
            if pokemon.id == Makuhita or pokemon.id == Hariyama:
                if pokemon.id == Hariyama:
                    score += 1
                if energy_count < 3:
                    score += 100
                if attacker2:
                    score -= 50
            elif pokemon.id == Lunatone:
                score -= 100
            elif pokemon.id == Solrock:
                if energy_count < 1:
                    score += 20
                else:
                    score -= 100
            elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
                if pokemon.id == Mega_Lucario_ex:
                    score += 1
                if energy_count < 2:
                    score += 100
                if attacker1:
                    score -= 50
            return score

        scores = []
        for o in select.option:
            score = 0
            if o.type == OptionType.NUMBER:
                score = getattr(o, 'number', 0)
            elif o.type == OptionType.YES:
                score = 1
            elif o.type == OptionType.CARD:
                card = get_card(obs, getattr(o, 'area', AreaType.HAND), getattr(o, 'index', 0), getattr(o, 'playerIndex', my_index))
                if card != None:
                    energy_count = len(getattr(card, 'energies', []))
                    if context == SelectContext.SWITCH_ACTIVE_POKEMON or context == SelectContext.SETUP_ACTIVE_POKEMON:
                        if getattr(o, 'playerIndex', my_index) == my_index:
                            score += energy_count * 2
                            if getattr(o, 'index', 0) == plan.attacker - 1:
                                score += 100
                            if card.id == Mega_Lucario_ex:
                                if my_prize == 2 or my_prize == 3:
                                    score += 8
                                else:
                                    score += 20
                            elif card.id == Hariyama and energy_count >= 2:
                                score += 15
                            elif card.id == Makuhita and energy_count >= 2:
                                score += 10
                            elif card.id == Solrock:
                                score += 5
                            elif card.id == Riolu:
                                score += 4
                        else:
                            if getattr(o, 'index', 0) == plan.target - 1:
                                score += 100
                    elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                        if card.id == Solrock:
                            if state.firstPlayer == my_index:
                                score = 2
                            else:
                                score = 4
                        elif card.id == Riolu:
                            score = 3
                        elif card.id == Makuhita:
                            score = 1
                    elif context == SelectContext.SETUP_BENCH_POKEMON:
                        if card.id == Solrock:
                            if state.firstPlayer == my_index:
                                score = 2
                            else:
                                score = 4
                        elif card.id == Riolu:
                            score = 3
                        elif card.id == Makuhita:
                            score = 1
                    elif context == SelectContext.DISCARD:
                        # Prefer discarding cards we have excess of / least useful right now.
                        score = 100
                        if card.id == Basic_Fighting_Energy:
                            # Safe to shed extra energy once we have enough on field.
                            attached_energy = sum(
                                len(getattr(p, 'energies', [])) for p in (my_state.active + my_state.bench) if p is not None
                            )
                            score = 250 if attached_energy >= 4 else 20
                        elif card.id in (Mega_Lucario_ex, Riolu, Hariyama, Makuhita):
                            # Never volunteer to discard our attackers/evolution pieces.
                            score = -50
                        elif card.id in (Lillie_Determination, Carmine, Boss_Orders, Premium_Power_Pro):
                            # Keep key supporters unless we already hold duplicates.
                            score = 5 if hand_counts[card.id] <= 1 else 80
                        else:
                            score = 60
                    elif context == getattr(SelectContext, 'LOOK', -1):
                        # Used when searching/revealing cards (e.g. Dusk Ball, Poke Pad).
                        score = 10
                        if card.id == Mega_Lucario_ex:
                            score = 90 if field_counts[Riolu] >= 1 else 40
                        elif card.id == Riolu:
                            score = 70 if field_counts[Riolu] + field_counts[Mega_Lucario_ex] < 2 else 5
                        elif card.id in (Hariyama, Makuhita):
                            score = 55
                        elif card.id == Basic_Fighting_Energy:
                            score = 45
                        elif card.id in (Boss_Orders, Premium_Power_Pro, Carmine, Lillie_Determination):
                            score = 50
                    elif context == getattr(SelectContext, 'TO_HAND', -1):
                        score = 200 - hand_counts[card.id] * 100
                        if card.id == Makuhita:
                            if field_counts[card.id] >= 1:
                                score -= 10
                            else:
                                score += 10
                        elif card.id == Hariyama:
                            if field_counts[Makuhita] >= 1:
                                score += 20
                            else:
                                score -= 20
                        elif card.id == Lunatone:
                            if field_counts[card.id] >= 1:
                                score -= 250
                            else:
                                score += 60
                        elif card.id == Solrock:
                            if field_counts[card.id] >= 1:
                                score -= 250
                            else:
                                score += 50
                        elif card.id == Riolu:
                            if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                                score -= 150
                            elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                                score -= 3
                            else:
                                score += 40
                        elif card.id == Mega_Lucario_ex:
                            if field_counts[Riolu] >= 1:
                                score += 40
                            else:
                                score -= 15
                        elif card.id == Basic_Fighting_Energy:
                            if not ability_used or not state.energyAttached:
                                score += 30
                            else:
                                score -= 1
                    elif context == getattr(SelectContext, 'ATTACH_FROM', -1):
                        score = energy_score(card, o.area == AreaType.ACTIVE)
                    else:
                        # Generic fallback for any other CARD-select context not
                        # explicitly modeled above, so it isn't a blind pick-index-0.
                        score = 50
                        if card.id in (Mega_Lucario_ex, Riolu, Hariyama, Makuhita):
                            score += 30
            elif o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, getattr(o, 'index', 0), my_index)
                if card is None:
                    score = -1
                elif card.id in {673, 674, 675, 676, 677, 678}: # Pokemon IDs
                    score = W['play_pokemon_base']
                    if card.id == Lunatone or card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score = -1
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score = -1
                else:
                    score = 10000
                    deck_count = getattr(my_state, 'deckCount', 0)
                    is_desperate = (field_counts[Mega_Lucario_ex] == 0)

                    if card.id in (Dusk_Ball, Poke_Pad):
                        if deck_count < 15 and not is_desperate:
                            score = -1
                        else:
                            score = W['play_dusk_pad']
                    elif card.id == Switch:
                        if plan.attacker <= 0:
                            score = -1
                        else:
                            score = W['play_switch']
                    elif card.id == Premium_Power_Pro:
                        supporter_played = getattr(state, 'supporterPlayed', False)
                        if supporter_played and plan.remain_hp <= 0:
                            score = -1
                        elif not can_attack:
                            if not supporter_played and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                                score = 3050
                            else:
                                score = -1
                        else:
                            score = W['play_premium']
                    elif card.id == Boss_Orders:
                        if plan.target >= 1:
                            score = W['play_boss']
                        else:
                            score = -1
                    elif card.id == Carmine:
                        score = W['play_carmine'] if (deck_count >= 20 or is_desperate) else -1
                    elif card.id == Lillie_Determination:
                        score = W['play_lillie'] if (deck_count >= 15 or is_desperate) else -1
                    elif card.id == Gravity_Mountain:
                        if stadium_id == 0:
                            score = -1

                    # Universal Deck-out safety check for ALL non-Pokemon plays
                    if deck_count < 15 and score > 0 and not is_desperate and card.id not in (Boss_Orders, Switch, Hero_Cape, Basic_Fighting_Energy, Basic_Grass_Energy):
                        # Dusk Ball, Poke Pad, Premium Power Pro, etc.
                        score = -1
            elif o.type == OptionType.ATTACH:
                card = get_card(obs, AreaType.HAND, getattr(o, 'index', 0), my_index)
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                if card is not None and card.id == Hero_Cape:
                    score = W['attach_hero_cape']
                    if pokemon is not None and pokemon.id == Riolu:
                        score += 100
                    elif pokemon is not None and pokemon.id == Mega_Lucario_ex:
                        score += 200
                else:
                    if pokemon is not None:
                        score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                    else:
                        score = 0
                    if o.inPlayArea == AreaType.ACTIVE:
                        if plan.attacker == 0 and plan.energy:
                            score += 200
                    else:
                        if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                            score += 200
            elif o.type == OptionType.EVOLVE:
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score = W['evolve_base'] + len(getattr(pokemon, 'energies', []))
                if pokemon is not None and pokemon.id == Makuhita and plan.target == 0:
                    score = -1
            elif o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, getattr(o, 'index', 0), my_index)
                if card is not None and card.id == 1267:  # Lumiose City
                    score = 1
                else:
                    score = W['ability_base']
            elif o.type == OptionType.RETREAT:
                if plan.attacker >= 1:
                    score = W['retreat_base']
                else:
                    score = -1
            elif o.type == OptionType.ATTACK:
                score = W['attack_base']
                if plan.attack_index == 1:
                    if o.attackId == 983:  # Mega Brave
                        score += 100
                else:
                    if o.attackId != 983:
                        score += 100

            scores.append(score)

        desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
        
        # --- KAGGLE ENGINE BUG WORKAROUND ---
        # The Kaggle cabt engine has a fatal bug where attaching energy from the discard
        # using Mega Lucario ex's Aura Jab causes an infinite loop at Context 21,
        # leading to an INACTIVE timeout loss. 
        # To avoid this, we MUST return [] when selecting energies or targets for it.
        effect = getattr(select, 'effect', None)
        effect_id = -1
        if isinstance(effect, dict):
            effect_id = effect.get('id', -1)
        elif effect is not None:
            effect_id = getattr(effect, 'id', -1)
            
        if effect_id == 678:
            if getattr(select, 'minCount', 1) == 0:
                return []
            # If minCount > 0, we are forced to pick. The engine might crash, but we must return something.
        
        
        if context == SelectContext.MAIN:
            # 2-Ply Minimax Lookahead Override
            my_deck_list = my_deck if 'my_deck' in locals() else []
            override = _search_decide(obs, desc_indices, scores, my_deck_list, agent)
            if override is not None:
                # Force the overriding move to the front
                desc_indices.remove(override)
                desc_indices.insert(0, override)
                
            o = select.option[desc_indices[0]]

            if o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, getattr(o, 'index', 0), my_index)
                if card is not None and card.id == Lunatone:
                    ability_used = True
        return desc_indices[:select.maxCount]
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"Agent Error: {e}")
        select_data = obs_dict.get("select", {})
        if not select_data:
            # Fallback for turn 0
            file_path = "deck.csv"
            if not os.path.exists(file_path):
                file_path = "/kaggle_simulations/agent/" + file_path
            with open(file_path, "r") as file:
                csv = file.read().split("\n")
            my_deck = []
            for i in range(60):
                my_deck.append(int(csv[i]))
            return my_deck
        options = select_data.get("option", [])
        if options:
            return [0]
        return []
