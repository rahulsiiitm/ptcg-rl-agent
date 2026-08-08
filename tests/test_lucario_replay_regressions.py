from collections import defaultdict
from types import SimpleNamespace

from cg.api import AreaType, Card, OptionType, Pokemon, SelectContext
import src.agent.rule_based_lucario as lucario


def _pokemon(card_id: int, energy_count: int = 0, hp: int = 100, max_hp: int | None = None) -> Pokemon:
    energies = [object()] * energy_count
    return Pokemon(
        id=card_id,
        serial=card_id,
        hp=hp,
        maxHp=max_hp or hp,
        appearThisTurn=False,
        energies=energies,
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def _policy(me_active, me_bench, opponent_active, opponent_prizes=6):
    policy = lucario.LucarioPolicy.__new__(lucario.LucarioPolicy)
    policy.me = SimpleNamespace(
        active=[me_active], bench=me_bench, hand=[], discard=[],
        prize=[object()] * 6, deckCount=60,
    )
    policy.opponent = SimpleNamespace(
        active=[opponent_active], bench=[], handCount=0,
        prize=[object()] * opponent_prizes,
    )
    policy.my_index = 0
    policy.op_index = 1
    policy.context = SelectContext.MAIN
    policy.field_counts = defaultdict(int)
    policy.hand_counts = defaultdict(int)
    policy.discard_counts = defaultdict(int)
    policy.has_ready_lucario_line = False
    policy.has_ready_hariyama_line = False
    policy.my_prizes_left = len(policy.me.prize)
    return policy


def test_lunar_cycle_is_deferred_until_useful_manual_attachment(monkeypatch):
    """Regression for losses 90611080, 90612665, and 90621445."""
    ability = SimpleNamespace(type=OptionType.ABILITY, area=AreaType.ACTIVE, index=0)
    attach = SimpleNamespace(
        type=OptionType.ATTACH,
        index=0,
        inPlayArea=AreaType.BENCH,
        inPlayIndex=0,
    )
    lunatone = _pokemon(lucario.C.LUNATONE)
    riolu = _pokemon(lucario.C.RIOLU, energy_count=0)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=1, playerIndex=0)

    def fake_get_card(_obs, area, index, _player_index):
        if area == AreaType.ACTIVE:
            return lunatone
        if area == AreaType.HAND:
            return energy
        if area == AreaType.BENCH:
            return riolu
        return None

    monkeypatch.setattr(lucario, "get_card", fake_get_card)

    policy = lucario.LucarioPolicy.__new__(lucario.LucarioPolicy)
    policy.obs = object()
    policy.context = SelectContext.MAIN
    policy.state = SimpleNamespace(energyAttached=False)
    policy.select = SimpleNamespace(option=[ability, attach])
    policy.my_index = 0
    policy.hand_counts = defaultdict(int, {lucario.C.BASIC_FIGHTING_ENERGY: 1})
    policy._low_deck = lambda: False
    policy._score_attach = lambda _option: 12_000

    assert policy.must_attach_before_lunar_cycle()
    assert policy._score_ability(ability) == 11_999


def test_lunar_cycle_may_spend_energy_when_attackers_are_charged(monkeypatch):
    ability = SimpleNamespace(type=OptionType.ABILITY, area=AreaType.ACTIVE, index=0)
    attach = SimpleNamespace(
        type=OptionType.ATTACH,
        index=0,
        inPlayArea=AreaType.BENCH,
        inPlayIndex=0,
    )
    lunatone = _pokemon(lucario.C.LUNATONE)
    charged_lucario = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=1, playerIndex=0)

    def fake_get_card(_obs, area, index, _player_index):
        if area == AreaType.ACTIVE:
            return lunatone
        if area == AreaType.HAND:
            return energy
        if area == AreaType.BENCH:
            return charged_lucario
        return None

    monkeypatch.setattr(lucario, "get_card", fake_get_card)

    policy = lucario.LucarioPolicy.__new__(lucario.LucarioPolicy)
    policy.obs = object()
    policy.context = SelectContext.MAIN
    policy.state = SimpleNamespace(energyAttached=False)
    policy.select = SimpleNamespace(option=[ability, attach])
    policy.my_index = 0
    policy.hand_counts = defaultdict(int, {lucario.C.BASIC_FIGHTING_ENERGY: 1})
    policy._low_deck = lambda: False

    assert not policy.must_attach_before_lunar_cycle()
    assert policy._score_ability(ability) == lucario.W["ability_base"]


def test_end_beats_a_useless_switch():
    """Regression for 90647525: Switch promoted a zero-Energy Mega, then Ended."""
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, hp=200),
        [_pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340)],
        _pokemon(lucario.C.HYDRAPPLE_EX, energy_count=1, hp=330),
    )
    lucario.plan = lucario.AttackPlan()
    end = SimpleNamespace(type=OptionType.END)
    switch = Card(id=lucario.C.SWITCH, serial=1, playerIndex=0)

    assert policy._score_option(end) == 0
    assert policy._score_play_trainer(switch) < policy._score_option(end)


def test_terminal_ogerpon_threat_forces_single_prize_pivot():
    """Regression for 90636236: do not expose a 3-prize Mega at 10 HP."""
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=10, max_hp=340)
    hariyama = _pokemon(lucario.C.HARIYAMA, hp=200)
    ogerpon = _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, energy_count=10, hp=210)
    policy = _policy(mega, [hariyama], ogerpon, opponent_prizes=3)
    lucario.plan = lucario.AttackPlan()

    assert policy.terminal_defense_required()
    assert policy._safe_pivot_indices() == [0]
    retreat = SimpleNamespace(type=OptionType.RETREAT)
    assert policy._score_option(retreat) == 100_000


def test_switch_cannot_reexpose_a_doomed_mega_after_safe_retreat():
    """Regression for 90639619: Hariyama retreated safely, then Switch undid it."""
    hariyama = _pokemon(lucario.C.HARIYAMA, hp=200)
    doomed_mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=40, max_hp=340)
    froslass = _pokemon(lucario.C.MEGA_FROSLASS_EX, energy_count=3, hp=330)
    policy = _policy(hariyama, [doomed_mega], froslass, opponent_prizes=3)
    policy.me.hand = [object()] * 7
    lucario.plan = lucario.AttackPlan(attacker=1, target=0, attack_index=1, remain_hp=50)
    switch = Card(id=lucario.C.SWITCH, serial=1, playerIndex=0)

    assert not policy.terminal_defense_required()
    assert policy._pivot_threatened_prizes(0) == 3
    assert policy._score_play_trainer(switch) == -1


def test_game_winning_attack_overrides_terminal_retreat():
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=10, max_hp=340)
    target = _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, energy_count=10, hp=100)
    policy = _policy(mega, [_pokemon(lucario.C.HARIYAMA, hp=200)], target, opponent_prizes=2)
    lucario.plan = lucario.AttackPlan(attacker=0, target=0, attack_index=1, remain_hp=-170)

    attack = SimpleNamespace(type=OptionType.ATTACK, attackId=lucario.MEGA_BRAVE)
    retreat = SimpleNamespace(type=OptionType.RETREAT)
    assert policy._score_option(attack) > policy._score_option(retreat)


def test_dragapult_counts_active_and_bench_prizes_together():
    """Regression for 90637798: Phantom Dive can take multiple prizes at once."""
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=200, max_hp=340),
        [_pokemon(lucario.C.MAKUHITA, hp=60, max_hp=100)],
        _pokemon(lucario.C.DRAGAPULT_EX, energy_count=2, hp=320),
        opponent_prizes=4,
    )
    lucario.plan = lucario.AttackPlan()

    assert policy._threatened_prizes() == 4
    assert policy.terminal_defense_required()


def test_crustle_matchup_builds_non_ex_attacker_first():
    """Regression for 90652242: Energies cannot remain stranded on Pokemon ex."""
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340),
        [_pokemon(lucario.C.HARIYAMA, hp=200)],
        _pokemon(lucario.C.CRUSTLE, energy_count=3, hp=150),
    )

    mega_score = policy._energy_target_score(policy.me.active[0], active=True)
    hariyama_score = policy._energy_target_score(policy.me.bench[0], active=False)
    assert hariyama_score > mega_score


def test_crustle_energy_diversifies_after_hariyama_is_ready():
    """Regression for 90836708: a Bossed Mega needed one stored Energy."""
    ready_hariyama = _pokemon(lucario.C.HARIYAMA, energy_count=3, hp=80)
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=0, hp=340)
    backup_makuhita = _pokemon(lucario.C.MAKUHITA, energy_count=0, hp=80)
    policy = _policy(
        ready_hariyama,
        [mega, backup_makuhita],
        _pokemon(lucario.C.CRUSTLE, hp=150),
    )
    policy.has_ready_hariyama_line = True

    assert (
        policy._energy_target_score(mega, active=False)
        > policy._energy_target_score(backup_makuhita, active=False)
    )


def test_energy_is_not_attached_to_terminally_doomed_active(monkeypatch):
    # With zero Energy, one attachment still cannot pay Mega Lucario's
    # two-Energy retreat cost, so this is not an attach-then-retreat out.
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=0, hp=10, max_hp=340)
    policy = _policy(
        mega,
        [_pokemon(lucario.C.HARIYAMA, hp=200)],
        _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, energy_count=10, hp=210),
        opponent_prizes=3,
    )
    policy.obs = object()
    policy.state = SimpleNamespace(energyAttached=False)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=2, playerIndex=0)
    option = SimpleNamespace(
        type=OptionType.ATTACH, index=0,
        inPlayArea=AreaType.ACTIVE, inPlayIndex=0,
    )

    def fake_get_card(_obs, area, _index, _player_index):
        return energy if area == AreaType.HAND else mega

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    lucario.plan = lucario.AttackPlan()
    assert policy._score_attach(option) == -1


def test_overcharged_damaged_mega_does_not_take_energy_from_live_hariyama():
    """Regression for 90962113/90963811: stop stacking 4-8 Energy on Mega."""
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=4, hp=10, max_hp=340)
    hariyama = _pokemon(lucario.C.HARIYAMA, energy_count=0, hp=150)
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=120, max_hp=340),
        [mega, hariyama],
        _pokemon(lucario.C.ARCHALUDON_EX, energy_count=3, hp=300),
        opponent_prizes=1,
    )

    assert (
        policy._energy_target_score(hariyama, active=False)
        > policy._energy_target_score(mega, active=False)
    )


def test_terminal_state_allows_only_available_attack_attachment(monkeypatch):
    """Regression for 90961229/90963811: attack when no safe pivot exists."""
    active = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=0, hp=120, max_hp=340)
    doomed_bench = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=4, hp=10, max_hp=340)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=7, playerIndex=0)
    policy = _policy(
        active,
        [doomed_bench],
        _pokemon(lucario.C.ARCHALUDON_EX, energy_count=3, hp=300),
        opponent_prizes=1,
    )
    policy.obs = object()
    policy.state = SimpleNamespace(energyAttached=False)
    lucario.plan = lucario.AttackPlan(
        attacker=0, target=0, attack_index=0, remain_hp=170, needs_energy=True
    )
    option = SimpleNamespace(index=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)

    def fake_get_card(_obs, area, _index, _player_index):
        return energy if area == AreaType.HAND else active

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    assert policy.terminal_defense_required()
    assert policy._safe_pivot_indices() == []
    assert policy._score_attach(option) > 50_000


def test_alakazam_forces_lethal_mega_brave_over_aura_jab():
    """Regression for confirmed nonlethal Aura-over-Brave v19 losses."""
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=340)
    alakazam = _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=110, max_hp=140)
    policy = _policy(mega, [], alakazam, opponent_prizes=2)
    policy.state = SimpleNamespace(turn=10, energyAttached=True)
    policy.my_prizes_left = 4
    policy.can_switch = False
    policy.can_gust = False
    policy.can_use_mega_brave = True
    policy.hand_counts = defaultdict(int)
    policy.discard_counts = defaultdict(int, {lucario.C.BASIC_FIGHTING_ENERGY: 3})
    policy.opponent.handCount = 29

    policy._plan_attack()

    assert lucario.plan.attack_index == 1
    assert lucario.plan.remain_hp <= 0
    assert policy._opponent_has_terminal_threat()
    assert not policy.terminal_defense_required()


def test_uncharged_riolu_does_not_evolve_into_three_prize_alakazam_target(monkeypatch):
    riolu = _pokemon(lucario.C.RIOLU, energy_count=1, hp=80)
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, hp=150),
        [riolu],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
        opponent_prizes=3,
    )
    policy.obs = object()
    policy.opponent.handCount = 18
    option = SimpleNamespace(
        type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
        inPlayArea=AreaType.BENCH, inPlayIndex=0,
    )
    evolved = Card(id=lucario.C.MEGA_LUCARIO_EX, serial=4, playerIndex=0)

    def fake_get_card(_obs, area, _index, _player_index):
        return evolved if area == AreaType.HAND else riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    assert policy._score_evolve(option) == -1


def test_alakazam_matchup_powers_single_prize_hariyama_before_mega():
    mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340)
    hariyama = _pokemon(lucario.C.HARIYAMA, hp=150)
    policy = _policy(
        mega, [hariyama],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
        opponent_prizes=6,
    )
    assert (
        policy._energy_target_score(hariyama, active=False)
        > policy._energy_target_score(mega, active=True)
    )


def test_alakazam_charges_backup_makuhita_after_one_hariyama_is_ready():
    ready_hariyama = _pokemon(lucario.C.HARIYAMA, energy_count=3, hp=150)
    backup_makuhita = _pokemon(lucario.C.MAKUHITA, energy_count=0, hp=80)
    riolu = _pokemon(lucario.C.RIOLU, energy_count=0, hp=80)
    policy = _policy(
        ready_hariyama,
        [backup_makuhita, riolu],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
    )
    policy.has_ready_hariyama_line = True

    assert (
        policy._energy_target_score(backup_makuhita, active=False)
        > policy._energy_target_score(riolu, active=False)
    )


def test_one_energy_dragapult_is_treated_as_next_turn_phantom_dive():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=110, max_hp=340),
        [_pokemon(lucario.C.MAKUHITA, hp=60, max_hp=80)],
        _pokemon(lucario.C.DRAGAPULT_EX, energy_count=1, hp=320),
        opponent_prizes=4,
    )
    lucario.plan = lucario.AttackPlan()
    assert policy._threatened_prizes() == 4


def test_archaludon_attachment_is_included_in_terminal_threat():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=120, max_hp=340),
        [_pokemon(lucario.C.HARIYAMA, hp=150)],
        _pokemon(lucario.C.ARCHALUDON_EX, energy_count=2, hp=250, max_hp=300),
        opponent_prizes=3,
    )
    policy.opponent.bench = [_pokemon(lucario.C.RELICANTH, hp=100)]
    lucario.plan = lucario.AttackPlan()
    assert policy._visible_counterattack_damage() >= 220
    assert policy._opponent_has_terminal_threat()


def test_immediate_win_uses_our_remaining_prizes_not_opponents():
    """Regression for 90678340: a two-prize KO wins only when we need two."""
    target = _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, hp=80, max_hp=210)
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, energy_count=3, hp=150),
        [],
        _pokemon(lucario.C.ARCHALUDON_EX, hp=300),
        opponent_prizes=5,
    )
    policy.me.prize = [object()] * 2
    policy.my_prizes_left = 2
    policy.opponent.bench = [target]
    lucario.plan = lucario.AttackPlan(attacker=0, target=1, remain_hp=-130)

    assert policy._planned_immediate_win()

    policy.me.prize = [object()] * 3
    policy.my_prizes_left = 3
    assert not policy._planned_immediate_win()


def test_bench_lethal_requires_gust_before_attack_scores_as_win():
    target = _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, hp=80, max_hp=210)
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, energy_count=3, hp=150),
        [],
        _pokemon(lucario.C.ARCHALUDON_EX, hp=300),
        opponent_prizes=5,
    )
    policy.me.prize = [object()] * 2
    policy.my_prizes_left = 2
    policy.opponent.bench = [target]
    lucario.plan = lucario.AttackPlan(attacker=0, target=1, remain_hp=-130)

    attack = SimpleNamespace(type=OptionType.ATTACK, attackId=0)
    boss = Card(id=lucario.C.BOSS_ORDERS, serial=3, playerIndex=0)
    assert policy._score_play_trainer(boss) > policy._score_option(attack)


def test_rollout_does_not_corrupt_real_attack_plan(monkeypatch):
    """Regression for 90693686: simulated plans must not leak into prompts."""
    real_plan = lucario.AttackPlan(attacker=0, target=0, attack_index=1, remain_hp=-10)
    lucario.plan = real_plan

    class FakePolicy:
        def __init__(self, _obs):
            pass

        def rank_and_scores(self):
            lucario.plan = lucario.AttackPlan(attacker=2, target=3, remain_hp=99)
            return [0], [1.0]

    monkeypatch.setattr(lucario, "LucarioPolicy", FakePolicy)
    obs = SimpleNamespace(select=SimpleNamespace(option=[object()], maxCount=1))
    assert lucario._rollout_pick(obs) == [0]
    assert lucario.plan is real_plan


def test_one_energy_garchomp_is_a_next_turn_260_damage_threat():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=150, max_hp=340),
        [_pokemon(lucario.C.HARIYAMA, hp=200)],
        _pokemon(lucario.C.CYNTHIAS_GARCHOMP_EX, energy_count=1, hp=300, max_hp=330),
        opponent_prizes=3,
    )
    lucario.plan = lucario.AttackPlan()
    assert policy._visible_counterattack_damage() >= 260
    assert policy.terminal_defense_required()


def test_azumarill_tera_discount_and_psychic_weakness_are_visible():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340, max_hp=340),
        [_pokemon(lucario.C.HARIYAMA, hp=200)],
        _pokemon(lucario.C.AZUMARILL, energy_count=0, hp=120),
        opponent_prizes=3,
    )
    policy.opponent.bench = [
        _pokemon(lucario.C.TEAL_MASK_OGERPON_EX, hp=210)
    ]
    lucario.plan = lucario.AttackPlan()
    assert policy._visible_counterattack_damage() >= 460
    assert policy.terminal_defense_required()


def test_visible_backup_froslass_blocks_false_safe_active_ko():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=340),
        [_pokemon(lucario.C.HARIYAMA, hp=200)],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=100, max_hp=140),
        opponent_prizes=3,
    )
    policy.me.hand = [object()] * 7
    policy.opponent.handCount = 7
    policy.opponent.bench = [
        _pokemon(lucario.C.MEGA_FROSLASS_EX, energy_count=1, hp=310)
    ]
    lucario.plan = lucario.AttackPlan(attacker=0, target=0, attack_index=1, remain_hp=-170)

    assert not policy._planned_removes_active_threat()
    assert policy.terminal_defense_required()


def test_forced_promotion_prefers_candidate_that_does_not_lose_three_prizes():
    policy = _policy(
        None,
        [
            _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=120, max_hp=340),
            _pokemon(lucario.C.HARIYAMA, hp=200),
        ],
        _pokemon(lucario.C.CYNTHIAS_GARCHOMP_EX, energy_count=1, hp=300),
        opponent_prizes=3,
    )
    policy.context = SelectContext.TO_ACTIVE
    lucario.plan = lucario.AttackPlan()
    mega_option = SimpleNamespace(playerIndex=0, index=0)
    hariyama_option = SimpleNamespace(playerIndex=0, index=1)

    assert (
        policy._score_active_choice(hariyama_option, policy.me.bench[1])
        > policy._score_active_choice(mega_option, policy.me.bench[0])
    )


def test_grimmsnarl_shadow_bullet_counts_bench_prize():
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, hp=150, max_hp=200),
        [_pokemon(lucario.C.RIOLU, hp=20, max_hp=80)],
        _pokemon(lucario.C.GRIMMSNARL, energy_count=2, hp=320),
        opponent_prizes=2,
    )
    lucario.plan = lucario.AttackPlan()
    assert policy._threatened_prizes() == 2


def test_planned_lethal_attachment_beats_generic_riolu_priority(monkeypatch):
    solrock = _pokemon(lucario.C.SOLROCK, hp=110)
    riolu = _pokemon(lucario.C.RIOLU, hp=80)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=5, playerIndex=0)
    policy = _policy(solrock, [riolu], _pokemon(lucario.C.ARCHALUDON_EX, hp=70))
    policy.obs = object()
    policy.state = SimpleNamespace(energyAttached=False)
    policy.field_counts[lucario.C.LUNATONE] = 1
    lucario.plan = lucario.AttackPlan(attacker=0, target=0, remain_hp=0, needs_energy=True)

    active_attach = SimpleNamespace(index=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)
    bench_attach = SimpleNamespace(index=0, inPlayArea=AreaType.BENCH, inPlayIndex=0)

    def fake_get_card(_obs, area, _index, _player_index):
        if area == AreaType.HAND:
            return energy
        return solrock if area == AreaType.ACTIVE else riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    assert policy._score_attach(active_attach) > policy._score_attach(bench_attach)


def test_zero_energy_dragapult_cannot_claim_spread_prize_next_turn():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340, max_hp=340),
        [_pokemon(lucario.C.MAKUHITA, hp=60, max_hp=100)],
        _pokemon(lucario.C.DRAGAPULT_EX, energy_count=0, hp=320),
        opponent_prizes=1,
    )
    lucario.plan = lucario.AttackPlan()
    assert policy._threatened_prizes() == 0
    assert not policy.terminal_defense_required()


def test_backup_dragapult_successor_counts_bench_spread():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=250, max_hp=340),
        [_pokemon(lucario.C.MAKUHITA, hp=60, max_hp=100)],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=100, max_hp=140),
        opponent_prizes=1,
    )
    policy.opponent.bench = [
        _pokemon(lucario.C.DRAGAPULT_EX, energy_count=1, hp=320)
    ]
    lucario.plan = lucario.AttackPlan(attacker=0, target=0, remain_hp=-170)
    assert policy._successor_threatened_prizes() == 1
    assert policy.terminal_defense_required()


def test_terminal_attachment_that_enables_safe_retreat_is_prioritized(monkeypatch):
    solrock = _pokemon(lucario.C.SOLROCK, energy_count=0, hp=100, max_hp=110)
    safe_mega = _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=340)
    energy = Card(id=lucario.C.BASIC_FIGHTING_ENERGY, serial=6, playerIndex=0)
    policy = _policy(
        solrock,
        [safe_mega],
        _pokemon(lucario.C.CYNTHIAS_GARCHOMP_EX, energy_count=1, hp=330),
        opponent_prizes=1,
    )
    policy.obs = object()
    policy.state = SimpleNamespace(energyAttached=False)
    policy.me.asleep = False
    policy.me.paralyzed = False
    lucario.plan = lucario.AttackPlan()
    option = SimpleNamespace(index=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)

    def fake_get_card(_obs, area, _index, _player_index):
        return energy if area == AreaType.HAND else solrock

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    assert policy._score_attach(option) >= 100_000


def test_rollout_rank_exposes_top_k_but_pick_obeys_max_count(monkeypatch):
    real_plan = lucario.AttackPlan(attacker=0, target=0, remain_hp=-10)
    lucario.plan = real_plan

    class FakePolicy:
        def __init__(self, _obs):
            pass

        def rank_and_scores(self):
            lucario.plan = lucario.AttackPlan(attacker=3, target=3)
            return [2, 1, 0], [1.0, 2.0, 3.0]

    monkeypatch.setattr(lucario, "LucarioPolicy", FakePolicy)
    obs = SimpleNamespace(select=SimpleNamespace(option=[1, 2, 3], maxCount=1))
    assert lucario._rollout_pick(obs) == [2]
    assert lucario._rollout_rank(obs, 3) == [2, 1, 0]
    assert lucario.plan is real_plan


def test_lunatone_usage_tracks_search_final_choice(monkeypatch):
    remembered = []
    select = SimpleNamespace(option=[object(), object()], maxCount=1)
    obs = SimpleNamespace(
        select=select,
        current=SimpleNamespace(turn=12),
    )

    class FakePolicy:
        def __init__(self, _obs):
            self.context = SelectContext.MAIN
            self.select = select

        def rank_and_scores(self):
            return [0, 1], [30_000.0, 20_000.0]

        def _remember_lunatone_ability(self, ranked):
            remembered.append(list(ranked))

        def must_attach_before_lunar_cycle(self):
            return False

        def terminal_defense_required(self):
            return False

    monkeypatch.setattr(lucario, "LucarioPolicy", FakePolicy)
    monkeypatch.setattr(lucario, "to_observation_class", lambda _data: obs)
    monkeypatch.setattr(lucario, "_search_decide", lambda *_args: 1)
    lucario.pre_turn = -1

    assert lucario._base_lucario_agent({"select": {}}) == [1]
    assert remembered == [[1, 0]]


def test_hariyama_defensively_gusts_alakazam_into_harmless_abra():
    """Regression for 90715746 turn 4: preserve Hariyama for another turn."""
    hariyama = _pokemon(lucario.C.HARIYAMA, energy_count=1, hp=150)
    policy = _policy(
        hariyama,
        [],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
        opponent_prizes=5,
    )
    policy.opponent.handCount = 9
    policy.opponent.bench = [_pokemon(lucario.C.ABRA, hp=50)]
    policy.context = SelectContext.ACTIVATE
    policy.select = SimpleNamespace(
        contextCard=Card(id=lucario.C.HARIYAMA, serial=66, playerIndex=0),
        effect=None,
    )
    lucario.plan = lucario.AttackPlan()

    yes = SimpleNamespace(type=OptionType.YES)
    no = SimpleNamespace(type=OptionType.NO)
    assert policy._score_option(yes) > policy._score_option(no)
    assert policy._hariyama_gust_target() == 0


def test_second_hariyama_does_not_gust_alakazam_back_active():
    """Regression for 90715746: never undo the first defensive gust."""
    hariyama = _pokemon(lucario.C.HARIYAMA, energy_count=1, hp=150)
    policy = _policy(
        hariyama,
        [],
        _pokemon(lucario.C.ABRA, hp=50),
        opponent_prizes=5,
    )
    policy.opponent.handCount = 9
    policy.opponent.bench = [
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140)
    ]
    policy.context = SelectContext.ACTIVATE
    policy.select = SimpleNamespace(
        contextCard=Card(id=lucario.C.HARIYAMA, serial=65, playerIndex=0),
        effect=None,
    )
    lucario.plan = lucario.AttackPlan()

    yes = SimpleNamespace(type=OptionType.YES)
    no = SimpleNamespace(type=OptionType.NO)
    assert policy._score_option(no) > policy._score_option(yes)
    assert policy._hariyama_gust_target() is None


def test_safe_mega_evolves_before_lillie_shuffles_it_away(monkeypatch):
    """Regression for 90715746 turn 8: establish a durable attacker first."""
    riolu = _pokemon(lucario.C.RIOLU, energy_count=0, hp=80)
    policy = _policy(
        _pokemon(lucario.C.SOLROCK, energy_count=1, hp=110),
        [riolu],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=40, max_hp=140),
        opponent_prizes=3,
    )
    policy.opponent.handCount = 9
    policy.obs = object()
    mega = Card(id=lucario.C.MEGA_LUCARIO_EX, serial=76, playerIndex=0)
    lillie = Card(id=lucario.C.LILLIE_DETERMINATION, serial=106, playerIndex=0)
    evolve = SimpleNamespace(
        type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
        inPlayArea=AreaType.BENCH, inPlayIndex=0,
    )
    reset = SimpleNamespace(type=OptionType.PLAY, index=1)
    policy.select = SimpleNamespace(option=[evolve, reset])

    def fake_get_card(_obs, area, index, _player_index):
        if area == AreaType.HAND:
            return mega if index == 0 else lillie
        return riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    lucario.plan = lucario.AttackPlan()
    assert policy._score_evolve(evolve) > 0


def test_unsafe_mega_stays_unevolved_against_lethal_alakazam(monkeypatch):
    riolu = _pokemon(lucario.C.RIOLU, energy_count=0, hp=80)
    policy = _policy(
        _pokemon(lucario.C.SOLROCK, energy_count=1, hp=110),
        [riolu],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
        opponent_prizes=3,
    )
    policy.opponent.handCount = 18
    policy.obs = object()
    mega = Card(id=lucario.C.MEGA_LUCARIO_EX, serial=76, playerIndex=0)
    lillie = Card(id=lucario.C.LILLIE_DETERMINATION, serial=106, playerIndex=0)
    evolve = SimpleNamespace(
        type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
        inPlayArea=AreaType.BENCH, inPlayIndex=0,
    )
    policy.select = SimpleNamespace(
        option=[evolve, SimpleNamespace(type=OptionType.PLAY, index=1)]
    )

    def fake_get_card(_obs, area, index, _player_index):
        if area == AreaType.HAND:
            return mega if index == 0 else lillie
        return riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    lucario.plan = lucario.AttackPlan()
    assert policy._score_evolve(evolve) == -1


def test_powerful_hand_counts_alakazam_players_hand():
    """The attack's 'your hand' means the attacking player's hand."""
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340),
        [],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
    )
    policy.me.hand = [object()] * 7
    policy.opponent.handCount = 19

    assert policy._visible_counterattack_damage() == 380


def test_hand_trimmer_collapses_alakazam_hand_only():
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, energy_count=3, hp=150),
        [],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
    )
    policy.opponent.handCount = 19
    trimmer = Card(id=lucario.C.HAND_TRIMMER, serial=1, playerIndex=0)

    assert policy._score_play_trainer(trimmer) > 40_000

    policy.opponent.active = [_pokemon(lucario.C.CRUSTLE, hp=150)]
    assert policy._score_play_trainer(trimmer) == -1


def test_hand_trimmer_reduces_our_froslass_damage_exposure():
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=2, hp=340),
        [],
        _pokemon(lucario.C.MEGA_FROSLASS_EX, energy_count=1, hp=310),
    )
    policy.me.hand = [object()] * 8
    trimmer = Card(id=lucario.C.HAND_TRIMMER, serial=1, playerIndex=0)

    assert policy._score_play_trainer(trimmer) > 40_000


def test_safe_one_energy_mega_evolves_against_alakazam(monkeypatch):
    """Do not repeat the five v25 Alakazam losses with no Mega evolution."""
    riolu = _pokemon(lucario.C.RIOLU, energy_count=1, hp=80)
    policy = _policy(
        _pokemon(lucario.C.SOLROCK, energy_count=1, hp=110),
        [riolu],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
    )
    policy.me.hand = [object()] * 4
    policy.obs = object()
    mega = Card(id=lucario.C.MEGA_LUCARIO_EX, serial=77, playerIndex=0)
    lillie = Card(id=lucario.C.LILLIE_DETERMINATION, serial=99, playerIndex=0)
    evolve = SimpleNamespace(
        type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
        inPlayArea=AreaType.BENCH, inPlayIndex=0,
    )
    play_lillie = SimpleNamespace(type=OptionType.PLAY, index=1)
    policy.select = SimpleNamespace(option=[evolve, play_lillie])

    def fake_get_card(_obs, area, index, _player_index):
        if area == AreaType.HAND:
            return lillie if index == 1 else mega
        return riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    lucario.plan = lucario.AttackPlan()
    assert policy._score_evolve(evolve) >= 25_000


def test_lillie_setup_is_not_suppressed_by_opponent_alakazam():
    policy = _policy(
        _pokemon(lucario.C.RIOLU, hp=80),
        [],
        _pokemon(lucario.C.ALAKAZAM, energy_count=1, hp=140),
    )
    policy.me.hand = [object()] * 3
    policy.me.deckCount = 30
    policy.my_prizes_left = 4
    lillie = Card(id=lucario.C.LILLIE_DETERMINATION, serial=1, playerIndex=0)

    assert policy._score_play_trainer(lillie) > 0


def test_early_abra_does_not_block_setup_draw_supporter():
    """Prepare normally before an actual Powerful Hand attacker is in play."""
    policy = _policy(
        _pokemon(lucario.C.RIOLU, hp=80),
        [],
        _pokemon(lucario.C.ABRA, hp=50),
    )
    policy.me.hand = [object()] * 3
    policy.me.deckCount = 30
    policy.my_prizes_left = 6
    lillie = Card(id=lucario.C.LILLIE_DETERMINATION, serial=1, playerIndex=0)

    assert policy._score_play_trainer(lillie) > 0


def test_low_deck_search_is_blocked_even_when_board_is_desperate():
    """Regression for the v25 deck-out losses 90723054/90801870/90836708."""
    policy = _policy(
        _pokemon(lucario.C.LUNATONE, hp=110),
        [],
        _pokemon(lucario.C.CRUSTLE, hp=20),
    )
    policy.me.deckCount = 9
    policy.my_prizes_left = 6
    search = Card(id=lucario.C.DUSK_BALL, serial=1, playerIndex=0)

    assert policy._low_deck()
    assert policy._score_deck_search(search) == -1


def test_low_deck_keeps_ready_active_instead_of_opening_boss_stall():
    """Regression for 90723054: attack now; do not Switch into a deck-out lock."""
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=4, hp=320, max_hp=340),
        [
            _pokemon(lucario.C.LUNATONE, hp=110),
            _pokemon(lucario.C.SOLROCK, energy_count=1, hp=110),
        ],
        _pokemon(lucario.C.DREEPY, energy_count=1, hp=70),
    )
    policy.me.deckCount = 3
    policy.my_prizes_left = 3
    policy.state = SimpleNamespace(turn=13, energyAttached=True)
    policy.can_switch = True
    policy.can_gust = False
    policy.can_use_mega_brave = True
    policy.field_counts[lucario.C.LUNATONE] = 1

    policy._plan_attack()

    assert lucario.plan.attacker == 0
    assert lucario.plan.remain_hp <= 0


def test_planned_terminal_attack_breaks_legal_order_tie():
    """Regression for 90730426: Mega Brave wins; Aura Jab does not."""
    policy = _policy(
        _pokemon(lucario.C.MEGA_LUCARIO_EX, energy_count=3, hp=50, max_hp=340),
        [],
        _pokemon(lucario.C.CYNTHIAS_GARCHOMP_EX, hp=230, max_hp=330),
    )
    policy.my_prizes_left = 2
    lucario.plan = lucario.AttackPlan(
        attacker=0, target=0, attack_index=1, remain_hp=-40
    )
    aura = SimpleNamespace(type=OptionType.ATTACK, attackId=982)
    brave = SimpleNamespace(type=OptionType.ATTACK, attackId=lucario.MEGA_BRAVE)

    assert policy._score_option(brave) > policy._score_option(aura)


def test_bench_snipe_defense_boosts_evolve_priority(monkeypatch):
    vulnerable_riolu = _pokemon(lucario.C.RIOLU, hp=80, energy_count=0)
    mega_lucario = _pokemon(lucario.C.MEGA_LUCARIO_EX, hp=340, energy_count=0)
    
    # 1. Test Starmie ex (50 bench snipe) against 80 HP Riolu (80 <= 50 + 30)
    policy = _policy(
        _pokemon(lucario.C.HARIYAMA, hp=150, energy_count=0),
        [vulnerable_riolu],
        _pokemon(lucario.C.MEGA_STARMIE_EX, hp=330, energy_count=2),
    )
    policy.obs = object()
    
    def fake_get_card(_obs, area, _index, _player_index):
        return mega_lucario if area == AreaType.HAND else vulnerable_riolu

    monkeypatch.setattr(lucario, "get_card", fake_get_card)
    lucario.plan = lucario.AttackPlan()
    
    evolve_option = SimpleNamespace(
        type=OptionType.EVOLVE,
        inPlayArea=AreaType.BENCH,
        inPlayIndex=0,
        area=AreaType.HAND,
        index=0,
    )
    
    # Needs to be > 200,000 to beat an active takes_prize attack
    assert policy._score_evolve(evolve_option) >= 250_000

