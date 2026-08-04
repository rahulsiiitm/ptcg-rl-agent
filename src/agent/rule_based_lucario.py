import os
import sys
from collections import defaultdict

# Add project root to sys.path if running outside kaggle
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class  # type: ignore
except ImportError:
    pass # Will be imported inside Kaggle

# Fetch card metadata database and create an ID-to-Card lookup table
try:
    all_card = all_card_data()
    card_table = {c.cardId:c for c in all_card}
except:
    card_table = {}

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
    else:
        return None

def prize_count(pokemon) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
            count -= 1
    return max(0, count)

def pokemon_score(pokemon) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    
    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score

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
                    card = get_card(obs, AreaType.HAND, o.index, my_index)
                    if card.id == Switch:
                        can_switch = True
                    elif card.id == Boss_Orders:
                        can_op_switch = True
                elif o.type == OptionType.EVOLVE:
                    card = get_card(obs, AreaType.HAND, o.index, my_index)
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
                            data = card_table[op_pokemon.id]
                            if data.weakness == EnergyType.FIGHTING:
                                damage *= 2
                            elif data.resistance == EnergyType.FIGHTING:
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
            energy_count = len(pokemon.energies)
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
                score = o.number
            elif o.type == OptionType.YES:
                score = 1
            elif o.type == OptionType.CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card != None:
                    energy_count = 0
                    if isinstance(card, Pokemon):
                        energy_count = len(card.energies)
                    if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                        if o.playerIndex == my_index:
                            score += energy_count * 2
                            if o.index == plan.attacker - 1:
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
                            if o.index == plan.target - 1:
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
                    elif context == SelectContext.TO_HAND:
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
                    elif context == SelectContext.ATTACH_FROM:
                        score = energy_score(card, o.area == AreaType.ACTIVE)
            elif o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                data = card_table[card.id]
                if data.cardType == CardType.POKEMON:
                    score = 20000
                    if card.id == Lunatone or card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score = -1
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score = -1
                else:
                    score = 10000
                    if card.id == Switch:
                        if plan.attacker <= 0:
                            score = -1
                        else:
                            score = 6000
                    elif card.id == Premium_Power_Pro:
                        if state.supporterPlayed and plan.remain_hp <= 0:
                            score = -1
                        elif not can_attack:
                            if not state.supporterPlayed and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                                score = 3050
                            else:
                                score = -1
                        else:
                            score = 5000
                    elif card.id == Boss_Orders:
                        if plan.target >= 1:
                            score = 3200
                        else:
                            score = -1
                    elif card.id == Carmine:
                        score = 3000
                    elif card.id == Lillie_Determination:
                        score = 3100
                    elif card.id == Gravity_Mountain:
                        if stadium_id == 0:
                            score = -1
            elif o.type == OptionType.ATTACH:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                if card.id == Hero_Cape:
                    score = 7000
                    if pokemon.id == Riolu:
                        score += 100
                    elif pokemon.id == Mega_Lucario_ex:
                        score += 200
                else:
                    score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                    if o.inPlayArea == AreaType.ACTIVE:
                        if plan.attacker == 0 and plan.energy:
                            score += 200
                    else:
                        if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                            score += 200
            elif o.type == OptionType.EVOLVE:
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score = 9000 + len(pokemon.energies)
                if pokemon.id == Makuhita and plan.target == 0:
                    score = -1
            elif o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, o.index, my_index)
                if card.id == 1267:  # Lumiose City
                    score = 1
                else:
                    score = 30000
            elif o.type == OptionType.RETREAT:
                if plan.attacker >= 1:
                    score = 2000
                else:
                    score = -1
            elif o.type == OptionType.ATTACK:
                score = 1000
                if plan.attack_index == 1:
                    if o.attackId == 983:  # Mega Brave
                        score += 100
                else:
                    if o.attackId != 983:
                        score += 100

            scores.append(score)

        desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
        if context == SelectContext.MAIN:
            o = select.option[desc_indices[0]]
            if o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, o.index, my_index)
                if card.id == Lunatone:
                    ability_used = True
        return desc_indices[:select.maxCount]
    except Exception as e:
        print(f"Agent Error: {e}")
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
