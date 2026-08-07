from collections import defaultdict
from types import SimpleNamespace

from cg.api import AreaType, Card, OptionType, Pokemon, SelectContext
import src.agent.rule_based_lucario as lucario


def _pokemon(card_id: int, energy_count: int = 0) -> Pokemon:
    energies = [object()] * energy_count
    return Pokemon(
        id=card_id,
        serial=card_id,
        hp=100,
        maxHp=100,
        appearThisTurn=False,
        energies=energies,
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


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
