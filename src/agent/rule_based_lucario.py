from __future__ import annotations

import json
import os
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    KYOGRE = 721
    SNOVER = 722
    MEGA_ABOMASNOW_EX = 723

    ABRA = 741
    KADABRA = 742
    ALAKAZAM = 743

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678
    DWEBBLE = 344
    CRUSTLE = 345

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10

_ABRA_BONUS = 400
_KADABRA_BONUS = 400

# ---- Tunable priority weights (defaults reproduce the hand-tuned baseline) ----
# Keys match MUTATE_KEYS in evolve_lucario.py exactly. If you add a new
# scored branch and want it evolvable, add its key here AND to MUTATE_KEYS.
WEIGHTS = {
    "play_pokemon_base": 20000,
    "play_dusk_pad": 10000,     # Dusk Ball / Poke Pad / Fighting Gong (deck-out gated separately)
    "play_switch": 6000,
    "play_premium": 5000,       # Premium Power Pro when already attacking
    "play_boss": 3200,
    "play_carmine": 3000,
    "play_lillie": 3100,
    "attach_hero_cape": 7000,
    "evolve_base": 9000,
    "ability_base": 30000,
    "retreat_base": 2000,
    "attack_base": 1000,
}

# Load runtime overrides written by evolve_lucario.py. Checked in this order
# so both local dev and the packaged Kaggle submission resolve it.
for _p in ("lucario_w.json", "./lucario_w.json", "/kaggle_simulations/agent/lucario_w.json"):
    if os.path.exists(_p):
        try:
            WEIGHTS.update(json.load(open(_p)))
        except Exception:
            pass
        break

W = WEIGHTS


BASE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "main.py")))
DECK_PATH = os.path.join(BASE_DIR, "deck.csv")
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
if not os.path.exists(DECK_PATH):
    DECK_PATH = "deck.csv"
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    player = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return player.hand[index]
        case AreaType.DISCARD:
            return player.discard[index]
        case AreaType.ACTIVE:
            return player.active[index]
        case AreaType.BENCH:
            return player.bench[index]
        case AreaType.PRIZE:
            return player.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.ABRA:
        score += _ABRA_BONUS
    elif pokemon.id == C.KADABRA:
        score += _KADABRA_BONUS
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    if pokemon.id == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked[: self.select.maxCount]

    def rank_and_scores(self) -> tuple[list[int], list[float]]:
        """Like choose(), but returns the full ranking + raw scores instead of
        just the top-maxCount pick. Used by the search layer to build its
        candidate set and by greedy rollouts during search."""
        if not self.select.option or self.select.maxCount == 0:
            return [], []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked, scores

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has_crustle_axis(self) -> bool:
        return any(
            pokemon is not None and pokemon.id in {C.DWEBBLE, C.CRUSTLE}
            for pokemon in self._opponent_board()
        )

    def _opponent_is_water_deck(self) -> bool:
        return any(
            pokemon is not None and pokemon.id in {C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX}
            for pokemon in self._opponent_board()
        )

    def _should_preserve_hariyama(self) -> bool:
        return (
            self._opponent_has_crustle_axis()
            and self.hand_counts[C.HARIYAMA] >= 1
            and any(pokemon is not None and pokemon.id == C.MAKUHITA for pokemon in self._my_board())
        )

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self.my_prizes_left in {2, 3}:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    if my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == C.CRUSTLE:
                        damage = 0
                    else:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif op_data.resistance == EnergyType.FIGHTING:
                            damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            score += 100 if energy_count < 3 else 0
            score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 2500  # Firmly below Lucario line — don't waste energy here
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -2500
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            # Core win condition — always gets energy first
            score += 2000
            score += 500 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 500 if energy_count < 2 else 0
            score -= 200 if self.has_ready_lucario_line else 0
            if active and energy_count < 3:
                score += 800
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            if plan.attacker < 1:
                return -1
            # Penalize retreat proportional to attached energy being lost
            active_list = self.me.active or []
            energy_cost = len(active_list[0].energies) if active_list and active_list[0] else 0
            return W["retreat_base"] - (energy_cost * 1500)
        if option.type == OptionType.ATTACK:
            base = W["attack_base"]
            # FIX: Replay analysis found 174 cases of attack being skipped when Lucario
            # was ready. Boost attack score massively when Mega Lucario has 2+ energies
            # so it always executes last (highest priority = plays last = ends turn correctly).
            active_list = self.me.active or []
            if active_list:
                active_poke = active_list[0]
                if (active_poke.id == C.MEGA_LUCARIO_EX
                        and len(active_poke.energies) >= 2
                        and plan.attacker == 0):
                    base += 40_000

            # Scale attack urgency when behind on prizes
            my_prizes = self.my_prizes_left
            op_prizes = len(self.opponent.prize)
            prize_diff = op_prizes - my_prizes  # Negative means we are trailing
            if prize_diff < 0:
                base += abs(prize_diff) * 5_000  # Aggressively push KOs when behind
            elif prize_diff > 0:
                base += prize_diff * 1_000

            return base + 100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else base
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        # ---- ENGINE BUG WORKAROUND ----
        # Detect if this card's serial is physically in the discard pile.
        # If so, the engine has duplicated it on the bench. Selecting it crashes cabt.
        discard_serials = {d.serial for d in self.me.discard if hasattr(d, "serial")}
        if hasattr(card, "serial") and card.serial in discard_serials:
            return -10000
        # -------------------------------

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 20
        if option.index == plan.attacker - 1:
            score += 1000
        if card.id == C.MEGA_LUCARIO_EX:
            score += 80 if self.my_prizes_left in {2, 3} else 200
        elif card.id == C.HARIYAMA:
            score += 150
        elif card.id == C.MAKUHITA:
            score += 100
        elif card.id == C.SOLROCK:
            score += 50
        elif card.id == C.RIOLU:
            score += 40
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.MAKUHITA:
            score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = W["play_pokemon_base"]
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return W["play_switch"] if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            return W["play_premium"]
        if card.id == C.BOSS_ORDERS:
            base_boss = W["play_boss"] if plan.target >= 1 else -1
            # FIX: Replay analysis found 16 cases where Boss Orders was in hand but not
            # played despite opponent having a low-HP benched target. Dynamically boost
            # when any opponent bench pokemon is below 40% HP — it's a near-free KO.
            if base_boss > 0:
                for op_poke in (self.opponent.bench or []):
                    if op_poke is not None and op_poke.maxHp > 0:
                        if op_poke.hp / op_poke.maxHp < 0.4:
                            base_boss += 47_000  # Force it to top priority
                            break
            return base_boss
        if card.id == C.CARMINE:
            if self._should_preserve_hariyama():
                return -1
            if self._low_deck():
                return -1
            # Carmine draws 8 after discard — better with larger hand
            hand_size = len(self.me.hand)
            if hand_size >= 5:
                return W["play_carmine"] + (hand_size - 4) * 400
            return W["play_carmine"]
        if card.id == C.LILLIE_DETERMINATION:
            if self._low_deck():
                return -1
            # Lillie draws to 3 cards — penalize if hand is already big
            hand_size = len(self.me.hand)
            if hand_size >= 6:
                return W["play_lillie"] - (hand_size - 5) * 600
            return W["play_lillie"]
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        if card.id in {C.DUSK_BALL, C.POKE_PAD, C.FIGHTING_GONG}:
            return self._score_deck_search(card)
        return W["play_dusk_pad"]

    def _score_deck_search(self, card: Card) -> float:
        # Dusk Ball / Poke Pad / Fighting Gong all thin the deck. Never let
        # them outrank other plays unconditionally once deck-out is a risk.
        is_desperate = self.field_counts[C.MEGA_LUCARIO_EX] == 0 and self.field_counts[C.RIOLU] == 0
        if self._low_deck() and not is_desperate:
            return -1
        return W["play_dusk_pad"]

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and card_table[pokemon.id].stage2 for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        # Dynamic deck-out margin: how many more draws can we safely spend
        # before running out, given prizes still owed. Falls back to the
        # flat LOW_DECK_COUNT if we're already in a lethal position (no
        # need to hoard resources if the game is basically over).
        safe_draws = self.me.deckCount - self.my_prizes_left - 1
        return safe_draws < LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            score = W["attach_hero_cape"]
            if self._opponent_is_water_deck():
                if pokemon.id == C.RIOLU:
                    return 12200
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    return 12800
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
                # FIX: Replay analysis found 9 cases where Hero Cape was equipped AFTER
                # Lucario had already taken heavy damage (50%+ lost). Proactively boost
                # when Lucario is active and at full/high HP so it gets equipped early.
                if pokemon.maxHp > 0 and pokemon.hp / pokemon.maxHp > 0.70:
                    score += 8_000  # Equip early for full HP buffer benefit
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        evolved = get_card(self.obs, option.area, option.index, self.my_index)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not (
            evolved is not None and evolved.id == C.HARIYAMA and board_index == plan.attacker
        ):
            return -1
        return W["evolve_base"] + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        return W["ability_base"]

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


# ==================== 2-PLY SEARCH LAYER ====================
# Heuristic scoring above gets you a strong single-ply agent. This adds a
# shallow lookahead on top: for MAIN-phase decisions, sample the hidden
# information (opponent hand/deck, our own deck order), simulate a few
# candidate moves forward through our own turn plus the opponent's likely
# reply, and only override the heuristic's top pick if search shows a real
# margin. Falls back to pure heuristic silently if the engine's search API
# isn't available or anything goes wrong — this is a bonus layer, never a
# dependency.
import time
import random
import sys
from collections import Counter

try:
    from cg.api import search_begin, search_step, search_end  # type: ignore
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False

USE_SEARCH = True
N_DET = 3                  # hidden-information determinizations (was 2, v15 uses 3)
K_OPP = 3                  # opponent replies at ply-2 (was 2, v15 uses 3)
MAX_SUBSTEPS = 40
TIME_BUDGET_S = 0.80
SEARCH_MAX_OPTS = 24
DUMMY_BASIC = C.RIOLU
DUMMY_ENERGY = C.BASIC_FIGHTING_ENERGY

# ---- Opponent archetype belief model (ported from v15) ----
# Load deck templates from top20_decks/ folder for accurate hidden sampling.
_TEMPLATES = []  # list of (name, Counter, list[int])
_template_search_dirs = [
    "/kaggle_simulations/agent/top20_decks",
    "top20_decks",
    os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "main.py"))), "top20_decks"),
]
for _d in _template_search_dirs:
    if os.path.isdir(_d):
        for _fn in sorted(os.listdir(_d)):
            if not _fn.endswith(".csv"):
                continue
            try:
                with open(os.path.join(_d, _fn)) as _tf:
                    _ids = [int(x) for x in _tf.read().split() if x.strip()][:60]
                if len(_ids) == 60:
                    _TEMPLATES.append((_fn, Counter(_ids), _ids))
            except Exception:
                pass
        break

_CARD_DATA_LOOKUP = {c.cardId: c for c in all_card}


def _pokemon_ids_from_counter(counter):
    return {cid for cid in counter
            if _CARD_DATA_LOOKUP.get(cid)
            and _CARD_DATA_LOOKUP[cid].cardType == CardType.POKEMON}


_TEMPLATE_SIG = [(n, _pokemon_ids_from_counter(c), c, ids) for (n, c, ids) in _TEMPLATES]


def _match_archetype(op_seen):
    """Pick template with highest Pokémon-ID overlap vs opponent's visible cards."""
    op_mons = {cid for cid in op_seen
               if _CARD_DATA_LOOKUP.get(cid)
               and _CARD_DATA_LOOKUP[cid].cardType == CardType.POKEMON}
    if not op_mons or not _TEMPLATE_SIG:
        return None
    best, best_n = None, 0
    for name, sig, cnt, ids in _TEMPLATE_SIG:
        n = len(sig & op_mons)
        if n > best_n:
            best, best_n = (cnt, ids), n
    return best if best_n >= 1 else None


_search_ok = _SEARCH_IMPORT_OK
_search_reported = False


def _my_visible(state, me_i):
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
        for c in p.energyCards:
            seen[c.id] += 1
        for c in p.tools:
            seen[c.id] += 1
        for c in getattr(p, "preEvolution", []):
            seen[c.id] += 1
    if state.stadium and state.stadium[0].playerIndex == me_i:
        seen[state.stadium[0].id] += 1
    return seen


def _op_visible(state, op_i):
    op = state.players[op_i]
    seen = Counter()
    etype = Counter()
    for c in op.discard:
        seen[c.id] += 1
    for p in op.active + op.bench:
        if p is None:
            continue
        seen[p.id] += 1
        for c in p.energyCards:
            seen[c.id] += 1
        for c in p.tools:
            seen[c.id] += 1
        for c in getattr(p, "preEvolution", []):
            seen[c.id] += 1
        for e in getattr(p, "energies", []):
            etype[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_i:
        seen[state.stadium[0].id] += 1
    for c in op.prize:
        if c is not None:
            seen[c.id] += 1
    return seen, etype


def _sample_hidden(state, me_i):
    """One determinization of hidden zones.
    Our side: exact (we know our decklist).
    Opponent side: archetype-matched template if possible, else naive fallback.
    Ported from v15 for significantly more accurate MCTS."""
    me = state.players[me_i]
    op_i = 1 - me_i
    op = state.players[op_i]

    # --- Our deck + facedown prizes ---
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

    # --- Opponent hidden (deck + prize + hand): archetype-matched template ---
    op_seen, etype = _op_visible(state, op_i)
    tpl = _match_archetype(op_seen)
    if tpl is not None:
        cnt, _ = tpl
        pool = []
        for cid, n in cnt.items():
            pool.extend([cid] * max(0, n - op_seen.get(cid, 0)))
    else:
        # Fallback: use visible card distribution + dominant energy type
        etop = max(etype.items(), key=lambda x: x[1])[0] if etype else int(EnergyType.FIGHTING)
        energy_id = etop if 1 <= etop <= 8 else DUMMY_ENERGY
        top_card = max(op_seen.items(), key=lambda x: x[1])[0] if op_seen else None
        pool = ([top_card] * 30 if top_card else []) + [energy_id] * 30 + [DUMMY_BASIC] * 8

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
    if state is None:
        return 0.0
    if getattr(state, "result", None) is not None and state.result >= 0:
        if state.result == me_i:
            return 1e7
        if state.result == 2:
            return 0.0
        return -1e7
    me = state.players[me_i]
    op = state.players[1 - me_i]
    my_field = [p for p in (me.active + me.bench) if p]
    op_field = [p for p in (op.active + op.bench) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(getattr(p, "energies", [])) for p in my_field)
    op_en = sum(len(getattr(p, "energies", [])) for p in op_field)
    no_active = 0 if (me.active and me.active[0]) else 1
    lucario_bonus = sum(500 for p in my_field if getattr(p, "id", -1) == C.MEGA_LUCARIO_EX)
    return (
        1000.0 * (len(op.prize) - len(me.prize))
        + my_hp - op_hp
        + 5.0 * (my_en - op_en)
        - 4000.0 * no_active
        + lucario_bonus
    )


def _rollout_pick(obs):
    """Greedy pick used inside search rollouts (both our own continued turn
    and the opponent's simulated replies) — just the heuristic top choice."""
    try:
        ranked, _ = LucarioPolicy(obs).rank_and_scores()
    except Exception:
        n = len(getattr(obs.select, "option", []) or [])
        ranked = list(range(n))
    k = min(getattr(obs.select, "maxCount", 1) or 1, len(ranked)) if ranked else 0
    return ranked[:k] if k else []


def _greedy_complete_turn(sid, cur, owner, deadline):
    for _ in range(MAX_SUBSTEPS):
        if time.monotonic() > deadline:
            break
        cs = getattr(cur, "current", None)
        if cs is None or (getattr(cs, "result", None) is not None and cs.result >= 0):
            break
        if cs.yourIndex != owner or getattr(cur, "select", None) is None:
            break
        choice = _rollout_pick(cur)
        if not choice:
            break
        try:
            ss = search_step(sid, choice)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _advance_forced(sid, cur, owner, deadline, limit=8):
    for _ in range(limit):
        if time.monotonic() > deadline:
            break
        cs = getattr(cur, "current", None)
        sel = getattr(cur, "select", None)
        if (cs is None or sel is None or cs.yourIndex != owner
                or sel.context == SelectContext.MAIN
                or (getattr(cs, "result", None) is not None and cs.result >= 0)):
            break
        choice = _rollout_pick(cur)
        if not choice:
            break
        try:
            ss = search_step(sid, choice)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _search_decide(obs, base_order, base_scores):
    global _search_ok, _search_reported
    if not (USE_SEARCH and _search_ok):
        return None
    st = getattr(obs, "current", None)
    sel = getattr(obs, "select", None)
    if st is None or sel is None or sel.context != SelectContext.MAIN:
        return None
    n = len(sel.option)
    if n < 3 or n > SEARCH_MAX_OPTS or st.turn < 2:
        return None
    if getattr(obs, "search_begin_input", None) is None:
        if not _search_reported:
            _search_reported = True
        _search_ok = False
        return None

    me_i = st.yourIndex
    if not base_order:
        return None
    heur_top = base_order[0]
    cand = [heur_top]
    for i in base_order[1:]:
        if getattr(sel.option[i], "type", -1) in (OptionType.ATTACK, OptionType.END):
            continue
        if base_scores[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= 8:  # widened from 4 → 8 (v15 uses 8)
            break
    if len(cand) < 2:
        return None

    t0 = time.monotonic()
    deadline = t0 + TIME_BUDGET_S
    acc = {i: 0.0 for i in cand}
    n_eval = {i: 0 for i in cand}

    try:
        for _det in range(N_DET):
            if time.monotonic() > deadline:
                break
            hidden = _sample_hidden(st, me_i)
            try:
                ss0 = search_begin(obs, **hidden)
            except Exception:
                _search_ok = False
                return None
            root_sid = ss0.searchId

            for idx in cand:
                if time.monotonic() > deadline:
                    break
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid1, cur = ss.searchId, ss.observation
                sid1, cur = _greedy_complete_turn(sid1, cur, me_i, deadline)
                cs = getattr(cur, "current", None)
                if (cs is None or (getattr(cs, "result", None) is not None and cs.result >= 0)
                        or cs.yourIndex == me_i or getattr(cur, "select", None) is None):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue

                sid1, cur = _advance_forced(sid1, cur, 1 - me_i, deadline)
                cs = getattr(cur, "current", None)
                if (cs is None or getattr(cur, "select", None) is None
                        or cur.select.context != SelectContext.MAIN or cs.yourIndex == me_i):
                    acc[idx] += _leaf_eval(cs, me_i)
                    n_eval[idx] += 1
                    continue

                op_ch = _rollout_pick(cur)
                worst = None
                for k in range(min(K_OPP, len(op_ch))):
                    if time.monotonic() > deadline:
                        break
                    try:
                        ss2 = search_step(sid1, [op_ch[k]])
                    except Exception:
                        continue
                    sid2, cur2 = ss2.searchId, ss2.observation
                    sid2, cur2 = _greedy_complete_turn(sid2, cur2, 1 - me_i, deadline)
                    sid2, cur2 = _advance_forced(sid2, cur2, me_i, deadline, limit=6)
                    v = _leaf_eval(getattr(cur2, "current", None), me_i)
                    worst = v if worst is None else min(worst, v)

                if worst is None:
                    worst = _leaf_eval(cs, me_i)
                acc[idx] += worst
                n_eval[idx] += 1

            try:
                search_end()
            except Exception:
                pass

        n_top = n_eval.get(heur_top, 0)
        if n_top == 0:
            return None
        evaluated = [i for i in cand if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] + 1e-6 * base_scores[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        if best == heur_top:
            return None
        # Only override when search shows a real margin (~half a prize).
        if avg[best] < avg[heur_top] + 500.0:
            return None
        return best
    except Exception:
        return None
# ======================================================


def _safe_fallback(obs_dict: dict) -> list[int]:
    select_data = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not select_data:
        return list(my_deck)
    options = select_data.get("option", []) or []
    n = len(options)
    if n == 0:
        return []
    min_count = select_data.get("minCount", 1) or 1
    k = min(max(1, min_count), n)
    return list(range(k))


def agent(obs_dict: dict) -> list[int]:
    global pre_turn
    global ability_used
    global plan

    try:
        if obs_dict.get("select") is None and "current" not in obs_dict:
            pre_turn = -1
            ability_used = False
            plan = AttackPlan()
            return my_deck

        obs = to_observation_class(obs_dict)
        if obs.select is None:
            pre_turn = -1
            ability_used = False
            plan = AttackPlan()
            return my_deck

        if pre_turn != obs.current.turn:
            pre_turn = obs.current.turn
            ability_used = False
            plan = AttackPlan()

        policy = LucarioPolicy(obs)
        ranked, scores = policy.rank_and_scores()
        policy._remember_lunatone_ability(ranked)

        if not ranked:
            return []

        if policy.context == SelectContext.MAIN:
            override = _search_decide(obs, ranked, scores)
            if override is not None:
                ranked = [override] + [i for i in ranked if i != override]

        return ranked[: policy.select.maxCount]
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent Error: {e}")
        return _safe_fallback(obs_dict)

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
