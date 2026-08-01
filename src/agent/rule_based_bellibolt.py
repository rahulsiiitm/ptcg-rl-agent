"""
src/agent/rule_based_bellibolt.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 bespoke rule-based agent for the Iono's Bellibolt ex / Kilowattrel deck.

Priority order (§3 of spec):
  §3.1  Turn-0 / mulligan / setup  — return deck or place active
  §3.2  Setup bench                — place all Basics from hand
  §3.3  Attacker selection         — Crustle immunity branch first, then Bellibolt
  §3.4  Energy / Ability           — use Electric Streamer every turn
  §3.5  Supporter priority         — Boss > Cheren
  §3.6  Item priority              — Ultra Ball > Buddy-Buddy Poffin > Master Ball > Catcher
  §3.7  Retreat / Switch           — retreat Bellibolt on cooldown, switch in Kilowattrel
  §3.8  Attack                     — Thunderous Bolt or Mach Bolt; never END if attack legal
  §3.8.4 Legal fallback             — always return a legal index, never crash

Bellibolt cooldown (§4):
  Thunderous Bolt says "During your next turn, this Pokémon can't use attacks."
  The cabt engine enforces this by simply NOT presenting Thunderous Bolt as a legal
  option on that next turn — so we do NOT need to track cooldown ourselves. If no
  attack option whose name matches "Thunderous Bolt" is available on Bellibolt's
  turn, we treat it as being in cooldown and switch to Kilowattrel.

Crustle detection (§3.3 step 1):
  Crustle (any Crustle variant) blocks damage from ex Pokémon. The obs_dict surfaces
  the opponent's active Pokémon via obs_dict['current']['players'][opp_idx]['active'][0].
  We detect Crustle by checking the card 'id' against known Crustle IDs (hardcoded from
  EN_Card_Data.csv) or by checking the 'name' field for 'Crustle' (case-insensitive).

OPT TYPE CONSTANTS (from cabt API docs):
  SelectType.MAIN     = 0
  SelectType.CARD     = 1
  SelectType.YES_NO   = 9
  Option.type values for MAIN:
    ATTACK  = 13
    EVOLVE  = 9
    PLAY    = 7   (play a card from hand — covers supporters, items)
    ATTACH  = 8
    ABILITY = 10
    END     = 14
    RETREAT = 11
    SWITCH  = 12  (possibly, used when resolving switch targets)
"""

import os
import random

# ─── Card ID Constants ────────────────────────────────────────────────────────
TADBULB_ID        = 268
BELLIBOLT_EX_ID   = 269
WATTREL_ID        = 270
KILOWATTREL_ID    = 271

BUDDY_BUDDY_ID    = 1086
ULTRA_BALL_ID     = 1121
SWITCH_ID         = 1123
CATCHER_ID        = 1124
MASTER_BALL_ID    = 1125
LOVE_BALL_ID      = 1083
BOSS_ORDERS_ID    = 1182
CHEREN_ID         = 1224
AIR_BALLOON_ID    = 1174
LIGHTNING_ENERGY_ID = 4

# Crustle card IDs from EN_Card_Data.csv (both variants in Standard)
CRUSTLE_IDS       = {348, 349}   # add more if found

# ─── Option type constants from cabt API ─────────────────────────────────────
OPT_ATTACK  = 13
OPT_EVOLVE  = 9
OPT_PLAY    = 7
OPT_ATTACH  = 8
OPT_ABILITY = 10
OPT_END     = 14

SELECT_MAIN   = 0
SELECT_CARD   = 1   # engine asks us to pick a card from a list
SELECT_YES_NO = 9

# ─── Integer context codes (real Kaggle cabt engine sends ints, not strings) ──
# Mapped from observed episode JSONs + cabt docs
CTX_SETUP_ACTIVE   = 1    # choose starting active Pokémon
CTX_SETUP_BENCH    = 2    # choose bench Pokémon at game start
CTX_TO_HAND        = 4    # choose card to put in hand (ball search result)
CTX_DISCARD        = 5    # choose cards to discard (Ultra Ball cost)
CTX_EFFECT_TARGET  = 6    # choose effect target (Boss's Orders, Catcher)
CTX_SWITCH_IN      = 7    # choose Pokémon to switch into active
CTX_BENCH_PLACE    = 8    # place a card onto bench mid-game
CTX_DISCARD_ENERGY = 10   # choose energy to discard (Flashing Draw)
CTX_SWITCH_SELF    = 11   # choose own Pokémon to switch/retreat to
CTX_MAIN           = 0    # normal MAIN action turn
# String aliases (some older/local cabt versions use these)
_STR_CTX_SETUP_ACTIVE  = {"SETUPACTIVEPOKEMON", "SETUP_ACTIVE", "ISFIRST", "ISACTIVE"}
_STR_CTX_SETUP_BENCH   = {"SETUPBENCHPOKEMON", "SETUP_BENCH", "BENCHPOKEMON"}
_STR_CTX_TO_HAND       = {"TOHAND", "TO_HAND", "CARDSELECT", "SEARCHHAND"}
_STR_CTX_DISCARD       = {"DISCARD", "DISCARD_COST", "DISCARDCARD"}
_STR_CTX_EFFECT_TARGET = {"EFFECTTARGET", "EFFECT_TARGET", "BOSSTARGET", "GUSSTARGET"}
_STR_CTX_SWITCH_IN     = {"SWITCHIN", "SWITCH_IN", "TO_ACTIVE", "TOACTIVE", "SWITCH"}
_STR_CTX_BENCH_PLACE   = {"BENCHPLACE", "BENCH_PLACE", "PLAYTOBENCH"}
_STR_CTX_DISCARD_ENERGY= {"DISCARDENERGY", "DISCARD_ENERGY"}

# ─── File path helpers ────────────────────────────────────────────────────────
def _read_deck() -> list[int]:
    """Read deck.csv, skipping comment lines.  Works locally and on Kaggle."""
    for path in [
        "deck.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ]:
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            return [int(x) for x in lines[:60]]
    return []


# ─── obs_dict helpers ─────────────────────────────────────────────────────────
def _get_players(obs_dict: dict):
    """Return (me, opponent) PlayerState dicts."""
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
    """Return the card ID of the player's active Pokémon, or None."""
    active_list = player.get("active", [])
    if not active_list:
        return None
    slot = active_list[0]
    if isinstance(slot, dict):
        return slot.get("id")
    return None


def _active_name(player: dict) -> str:
    """Return the name of the player's active Pokémon (lower-cased), or ''."""
    active_list = player.get("active", [])
    if not active_list:
        return ""
    slot = active_list[0]
    if isinstance(slot, dict):
        return str(slot.get("name", "")).lower()
    return ""


def _active_hp(player: dict) -> int:
    """Remaining HP of the active Pokémon."""
    active_list = player.get("active", [])
    if not active_list:
        return 999
    slot = active_list[0]
    if isinstance(slot, dict):
        return int(slot.get("hp", 999))
    return 999


def _active_max_hp(player: dict) -> int:
    """Max HP of the active Pokémon."""
    active_list = player.get("active", [])
    if not active_list:
        return 999
    slot = active_list[0]
    if isinstance(slot, dict):
        return int(slot.get("maxHp", 999))
    return 999


def _bench_ids(player: dict) -> list[int]:
    """Card IDs of all bench Pokémon."""
    return [
        s.get("id") for s in player.get("bench", [])
        if isinstance(s, dict) and s.get("id") is not None
    ]


def _hand_ids(player: dict) -> list[int]:
    """Card IDs of cards in hand (may be visible or 0 if hidden)."""
    return [
        c.get("id") for c in player.get("hand", [])
        if isinstance(c, dict) and c.get("id") is not None
    ]


def _energy_count(player: dict) -> int:
    """Total Lightning energy attached to my active Pokémon."""
    active_list = player.get("active", [])
    if not active_list:
        return 0
    slot = active_list[0]
    if not isinstance(slot, dict):
        return 0
    return len([e for e in slot.get("energyCards", [])
                if isinstance(e, dict) and e.get("id") == LIGHTNING_ENERGY_ID])


def _is_crustle(opp: dict, obs_dict: dict = None) -> bool:
    """§3.3 step 1 — detect Crustle as the opponent's active Pokémon.
       Also handles Neutralization Zone (1247) which blocks all our Bellibolt ex attacks.
    """
    if obs_dict:
        stadium = obs_dict.get("current", {}).get("stadium", [])
        if stadium and isinstance(stadium, list) and len(stadium) > 0:
            if stadium[0].get("id") == 1247:
                return True
                
    active_id = _active_id(opp)
    if active_id in CRUSTLE_IDS:
        return True
    # Belt-and-suspenders: also check name
    name = _active_name(opp)
    return "crustle" in name


def _my_active_is_bellibolt(me: dict) -> bool:
    return _active_id(me) == BELLIBOLT_EX_ID


def _has_kilowattrel_on_bench(me: dict) -> bool:
    return KILOWATTREL_ID in _bench_ids(me)


# ─── Option search helpers ────────────────────────────────────────────────────
def _find_options_of_type(options: list, opt_type: int) -> list[tuple[int, dict]]:
    """Return list of (index, option) pairs matching the given opt_type."""
    return [(i, opt) for i, opt in enumerate(options)
            if isinstance(opt, dict) and opt.get("type") == opt_type]


def _opt_card_id(opt: dict) -> int | None:
    """Extract card ID from an option dict."""
    card = opt.get("card") or opt.get("target")
    if isinstance(card, dict):
        return card.get("id")
    return opt.get("cardId") or opt.get("id")


def _opt_target_id(opt: dict) -> int | None:
    """Extract target card ID (for Boss, Catcher etc.)."""
    target = opt.get("target") or opt.get("card")
    if isinstance(target, dict):
        return target.get("id")
    return None


def _opt_name(opt: dict) -> str:
    """Extract attack/ability name from option."""
    return str(opt.get("name", "") or opt.get("skillName", "")).lower()


def _opt_damage(opt: dict) -> int:
    """Extract expected damage from an attack option."""
    try:
        return int(opt.get("damage", 0) or 0)
    except (ValueError, TypeError):
        return 0


# ─── Sub-decision helpers ─────────────────────────────────────────────────────

def _pick_setup_active(options: list) -> list[int]:
    """
    §3.1 — When choosing a starting active Pokémon, prefer Tadbulb > Wattrel.
    Never start a Bellibolt ex (can't evolve-in as active starter anyway).
    """
    priority = [TADBULB_ID, WATTREL_ID, TADBULB_ID]  # just ordering preference
    for card_id in [TADBULB_ID, WATTREL_ID]:
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and _opt_card_id(opt) == card_id:
                return [i]
    # Fallback: first available
    return [0]


def _pick_setup_bench(options: list) -> list[int]:
    """
    §3.2 — Place ALL valid bench Pokémon. Prefer Tadbulbs first, then Wattrels.
    The engine's maxCount determines how many we can place this prompt.
    """
    preferred_order = [TADBULB_ID, WATTREL_ID, KILOWATTREL_ID, BELLIBOLT_EX_ID]
    selected = []
    remaining = list(range(len(options)))
    for card_id in preferred_order:
        for i in remaining:
            opt = options[i]
            if isinstance(opt, dict) and _opt_card_id(opt) == card_id:
                selected.append(i)
                remaining.remove(i)
                break
    # Fill remaining slots with whatever is available
    selected.extend(remaining)
    return selected


def _pick_attack(options: list, me: dict, opp: dict, obs_dict: dict) -> int | None:
    """
    §3.8 — Attack phase selection
      1. If opponent active is Crustle (or Neutralization Zone) → use Mach Bolt (on Kilowattrel bench swap first)
      2. If Bellibolt ex is active:
         a. Use Thunderous Bolt if available (not on cooldown)
         b. If Thunderous Bolt NOT available (cooldown) → signal to retreat instead
      3. If Kilowattrel is active → use Mach Bolt
    """
    attack_opts = _find_options_of_type(options, OPT_ATTACK)
    if not attack_opts:
        return None

    crustle_matchup = _is_crustle(opp, obs_dict)
    my_active_id = _active_id(me)

    if my_active_is_bellibolt := (my_active_id == BELLIBOLT_EX_ID):
        if crustle_matchup:
            # Bellibolt can't damage Crustle ex-immunity — do NOT attack, retreat instead
            return None

        # Bellibolt only has one attack. If the engine offers an attack, it is Thunderous Bolt.
        # (It hides the attack entirely if it is on cooldown).
        if attack_opts:
            return attack_opts[0][0]

        # Attack not offered -> cooldown -> retreat
        return None

    elif my_active_id == KILOWATTREL_ID:
        # Kilowattrel attacks regardless of Crustle — Mach Bolt bypasses ex-immunity
        return attack_opts[0][0]

    else:
        # Any other active Pokémon — use first available attack
        if attack_opts:
            return attack_opts[0][0]

    return None


def _should_retreat_for_kilowattrel(options: list, me: dict, opp: dict, obs_dict: dict) -> bool:
    """
    §3.7 — Retreat Bellibolt ex when:
      - It's on cooldown (no Thunderous Bolt offered by engine), AND
      - We have a Kilowattrel on bench to send in.
      - OR opponent is Crustle and Kilowattrel can answer it.
    """
    if not _my_active_is_bellibolt(me):
        return False
    if not _has_kilowattrel_on_bench(me):
        return False

    crustle_matchup = _is_crustle(opp, obs_dict)
    if crustle_matchup:
        return True

    # Check if Thunderous Bolt is unavailable (cooldown)
    # Since the engine doesn't provide attack names, we check if NO attacks are offered.
    attack_opts = _find_options_of_type(options, OPT_ATTACK)
    has_attack = len(attack_opts) > 0
    return not has_attack


def _pick_ability(options: list, me: dict) -> int | None:
    """
    §3.4 — Use Electric Streamer (Bellibolt Ability) if available.
    Electric Streamer attaches {L} energy from hand to any of our Pokémon.
    We should use this BEFORE playing supporters or attacking.
    """
    ability_opts = _find_options_of_type(options, OPT_ABILITY)
    for i, opt in ability_opts:
        if "streamer" in _opt_name(opt) or "electric" in _opt_name(opt):
            return i
    # Use any ability if we can't match by name
    if ability_opts:
        return ability_opts[0][0]
    return None


def _pick_play_card(options: list, me: dict, opp: dict, already_played_supporter: bool, obs_dict: dict) -> int | None:
    """
    §3.5 / §3.6 — Priority order for playing cards from hand:
    Boss's Orders > Ultra Ball > Buddy-Buddy Poffin > Master Ball > Cheren > Catcher > Switch > Air Balloon

    Boss's Orders: only if opponent has a weak/low-HP bench target.
    Cheren: draw to 6.
    """
    play_opts = _find_options_of_type(options, OPT_PLAY)
    if not play_opts:
        return None

    # Build a card_id → index map
    card_map: dict[int, int] = {}
    for i, opt in play_opts:
        cid = _opt_card_id(opt)
        if cid is not None and cid not in card_map:
            card_map[cid] = i

    # §3.5 Supporters (can only play one per turn)
    if not already_played_supporter:
        # Boss's Orders — pull a target to active to set up a KO
        if BOSS_ORDERS_ID in card_map:
            # Only play Boss if opponent has a bench (i.e., there's something to pull)
            opp_bench = opp.get("bench", [])
            if opp_bench:
                return card_map[BOSS_ORDERS_ID]

        # Cheren — draw cards; play if hand is running low
        if CHEREN_ID in card_map:
            my_hand_count = me.get("handCount", 10)
            if my_hand_count <= 5:
                return card_map[CHEREN_ID]

    # §3.6 Items (can play multiple per turn)
    # Ultra Ball — search for Bellibolt ex or Kilowattrel as needed
    if ULTRA_BALL_ID in card_map:
        bench = _bench_ids(me)
        # Fetch Bellibolt if we have a Tadbulb bench and no Bellibolt yet
        has_bellibolt = BELLIBOLT_EX_ID in bench
        has_kilowattrel = KILOWATTREL_ID in bench
        if not has_bellibolt or not has_kilowattrel:
            return card_map[ULTRA_BALL_ID]

    # Buddy-Buddy Poffin — bench Tadbulbs and Wattrels early
    if BUDDY_BUDDY_ID in card_map:
        bench = _bench_ids(me)
        bench_count = len([s for s in me.get("bench", []) if s])
        if bench_count < 4:  # still room on bench
            return card_map[BUDDY_BUDDY_ID]

    # Master Ball (ACE SPEC, 1 copy) — any Pokémon search
    if MASTER_BALL_ID in card_map:
        bench = _bench_ids(me)
        has_bellibolt = BELLIBOLT_EX_ID in bench
        has_kilowattrel = KILOWATTREL_ID in bench
        if not has_bellibolt or not has_kilowattrel:
            return card_map[MASTER_BALL_ID]

    # Love Ball — search for Pokémon with same name as one you have (Stage 1 search)
    if LOVE_BALL_ID in card_map:
        bench = _bench_ids(me)
        has_bellibolt = BELLIBOLT_EX_ID in bench
        has_kilowattrel = KILOWATTREL_ID in bench
        if not has_bellibolt or not has_kilowattrel:
            return card_map[LOVE_BALL_ID]

    # Catcher — bring up a low-HP opponent bench target
    if CATCHER_ID in card_map:
        opp_bench = opp.get("bench", [])
        for slot in opp_bench:
            if isinstance(slot, dict):
                hp = int(slot.get("hp", 999))
                max_hp = int(slot.get("maxHp", 999))
                # Bring up a damaged or low-HP target
                if hp <= 100 or (max_hp > 0 and hp / max_hp < 0.5):
                    return card_map[CATCHER_ID]

    # Cheren fallback — play anytime if we didn't play it above
    if not already_played_supporter and CHEREN_ID in card_map:
        return card_map[CHEREN_ID]

    # Switch — handled in retreat section, but allow if explicitly in PLAY options
    if SWITCH_ID in card_map:
        if _should_retreat_for_kilowattrel(options, me, opp, obs_dict):
            return card_map[SWITCH_ID]

    return None


def _pick_attach(options: list, me: dict) -> int | None:
    """
    §3.4 — Attach energy. Prefer attaching to Bellibolt ex active.
    If Bellibolt isn't active, attach to the lowest-energy Bellibolt on bench.
    """
    attach_opts = _find_options_of_type(options, OPT_ATTACH)
    if not attach_opts:
        return None
    # Just pick first option — engine only offers legal attaches
    # (manual-attach covers hand energy; Streamer ability handles rest)
    return attach_opts[0][0]


def _pick_evolve(options: list, me: dict) -> int | None:
    """
    §3.3 — Evolve priority:
      Tadbulb → Bellibolt ex (highest priority, don't evolve on T1 if just placed)
      Wattrel → Kilowattrel (secondary)
    We always evolve when legal (engine won't offer evolve on T1).
    """
    evolve_opts = _find_options_of_type(options, OPT_EVOLVE)
    if not evolve_opts:
        return None
    # Prefer evolving to Bellibolt ex first
    for i, opt in evolve_opts:
        target_id = _opt_target_id(opt) or _opt_card_id(opt)
        if target_id == BELLIBOLT_EX_ID:
            return i
    # Then Kilowattrel
    for i, opt in evolve_opts:
        target_id = _opt_target_id(opt) or _opt_card_id(opt)
        if target_id == KILOWATTREL_ID:
            return i
    # Fallback: any evolve
    return evolve_opts[0][0]


def _pick_retreat(options: list, me: dict) -> int | None:
    """
    §3.7 — Retreat option. Engine presents RETREAT as a MAIN option type.
    We pick retreat only when signalled by _should_retreat_for_kilowattrel.
    """
    # Retreat usually appears as type=11 or may be a PLAY card (Switch)
    retreat_opts = [(i, opt) for i, opt in enumerate(options)
                    if isinstance(opt, dict) and opt.get("type") == 11]
    if retreat_opts:
        return retreat_opts[0][0]
    return None


def _pick_switch_target(options: list, prefer_id: int = KILOWATTREL_ID) -> list[int]:
    """
    When prompted to pick a bench Pokémon to send in, prefer Kilowattrel.
    Used by the SWITCH SelectContext response.
    """
    for i, opt in enumerate(options):
        if isinstance(opt, dict) and _opt_card_id(opt) == prefer_id:
            return [i]
    return [0]


# ─── Main agent entrypoint ────────────────────────────────────────────────────

# Module-level state to detect if we played a supporter this turn.
# The engine calls agent() multiple times per "turn" (once per sub-decision).
# We reset at the start of each new turn number.
_state = {
    "last_turn": -1,
    "supporter_played_this_turn": False,
}


def _ctx_matches(ctx, int_code: int, str_set: set) -> bool:
    """Match select.context against both integer codes and legacy string names."""
    if isinstance(ctx, int):
        return ctx == int_code
    return str(ctx).upper().replace(' ', '_') in str_set


def _debug_log(step, ctx, stype, n_opts, result):
    """Append one line to agent_debug.log for post-mortem analysis."""
    try:
        with open("agent_debug.log", "a") as f:
            f.write(f"step={step} ctx={ctx} type={stype} n_opts={n_opts} -> {result}\n")
    except Exception:
        pass


def _handle_card_select(select_data: dict, me: dict, opp: dict) -> list[int]:
    """
    Handle ALL type=1 (Card-select) sub-prompts.
    The engine only offers legal cards — [0] is always safe, but we try to be smart.
    THIS is the function that was missing, causing INVALID at step 21.
    """
    options  = select_data.get("option", [])
    ctx      = select_data.get("context", "")
    max_count = select_data.get("maxCount", 1)

    if not options:
        return []

    # ── Setup active: choose starting active Pokémon ─────────────────────────
    if _ctx_matches(ctx, CTX_SETUP_ACTIVE, _STR_CTX_SETUP_ACTIVE):
        return _pick_setup_active(options)

    # ── Setup bench: place Basics on bench at game start ──────────────────────
    if _ctx_matches(ctx, CTX_SETUP_BENCH, _STR_CTX_SETUP_BENCH):
        picks = _pick_setup_bench(options)
        return picks[:max_count]

    # ── Bench placement mid-game (e.g. Buddy-Buddy Poffin resolves) ───────────
    if _ctx_matches(ctx, CTX_BENCH_PLACE, _STR_CTX_BENCH_PLACE):
        # Prefer Tadbulb > Wattrel, pick up to max_count
        priority = [TADBULB_ID, WATTREL_ID, KILOWATTREL_ID, BELLIBOLT_EX_ID]
        selected = []
        used = set()
        for pid in priority:
            for i, opt in enumerate(options):
                if i in used:
                    continue
                if isinstance(opt, dict) and _opt_card_id(opt) == pid:
                    selected.append(i)
                    used.add(i)
                    if len(selected) >= max_count:
                        return selected
        # Fill remaining slots
        for i in range(len(options)):
            if i not in used and len(selected) < max_count:
                selected.append(i)
        return selected if selected else [0]

    # ── Card search (Ultra Ball / Master Ball / Love Ball result) ─────────────
    if _ctx_matches(ctx, CTX_TO_HAND, _STR_CTX_TO_HAND):
        bench = _bench_ids(me)
        has_bellibolt   = BELLIBOLT_EX_ID in bench or _active_id(me) == BELLIBOLT_EX_ID
        has_kilowattrel = KILOWATTREL_ID   in bench or _active_id(me) == KILOWATTREL_ID
        search_priority = []
        if not has_bellibolt:
            search_priority.append(BELLIBOLT_EX_ID)
        if not has_kilowattrel:
            search_priority.append(KILOWATTREL_ID)
        search_priority += [TADBULB_ID, WATTREL_ID]
        for target_id in search_priority:
            for i, opt in enumerate(options):
                if isinstance(opt, dict) and _opt_card_id(opt) == target_id:
                    return [i]
        return [0]

    # ── Discard cost (Ultra Ball: discard 2 cards) ────────────────────────────
    if _ctx_matches(ctx, CTX_DISCARD, _STR_CTX_DISCARD):
        # Prefer to discard: energy first, then trainers, keep Pokémon last
        ENERGY_IDS  = {LIGHTNING_ENERGY_ID}
        POKEMON_IDS = {TADBULB_ID, WATTREL_ID, BELLIBOLT_EX_ID, KILOWATTREL_ID}
        tiers = [ENERGY_IDS, set(), POKEMON_IDS]  # energy > trainers > Pokémon
        selected = []
        used = set()
        # Tier 1: energy
        for i, opt in enumerate(options):
            if len(selected) >= max_count:
                break
            cid = _opt_card_id(opt) if isinstance(opt, dict) else None
            if cid in ENERGY_IDS and i not in used:
                selected.append(i)
                used.add(i)
        # Tier 2: non-Pokémon, non-energy (trainers/supporters)
        for i, opt in enumerate(options):
            if len(selected) >= max_count:
                break
            cid = _opt_card_id(opt) if isinstance(opt, dict) else None
            if cid not in ENERGY_IDS and cid not in POKEMON_IDS and i not in used:
                selected.append(i)
                used.add(i)
        # Tier 3: Pokémon (last resort)
        for i, opt in enumerate(options):
            if len(selected) >= max_count:
                break
            if i not in used:
                selected.append(i)
                used.add(i)
        return selected if selected else list(range(min(max_count, len(options))))

    # ── Effect target (Boss's Orders / Catcher — pick lowest HP bench) ─────────
    if _ctx_matches(ctx, CTX_EFFECT_TARGET, _STR_CTX_EFFECT_TARGET):
        best_i, best_hp = 0, 9999
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                hp = int(opt.get("hp", 9999))
                if hp < best_hp:
                    best_hp, best_i = hp, i
        return [best_i]

    # ── Switch/retreat target (choose bench Pokémon to send in) ───────────────
    if (_ctx_matches(ctx, CTX_SWITCH_IN, _STR_CTX_SWITCH_IN) or
            _ctx_matches(ctx, CTX_SWITCH_SELF, _STR_CTX_SWITCH_IN)):
        return _pick_switch_target(options)

    # ── Discard energy (Kilowattrel Flashing Draw cost) ───────────────────────
    if _ctx_matches(ctx, CTX_DISCARD_ENERGY, _STR_CTX_DISCARD_ENERGY):
        return [0]

    # ── Unknown Card-select context: always safe to return [0] ───────────────
    return [0]


def rule_based_bellibolt(obs_dict: dict) -> list[int]:
    """
    §3.8.4 Guaranteed legal-action contract:
      - Never returns an index out of bounds.
      - Never returns an empty list when options exist.
      - Never crashes — all exceptions caught, fallback to [0].

    FIX (v2): The original agent matched select.context against string literals
    only. The real Kaggle cabt engine sends INTEGER context codes (e.g. context=8
    for bench placement, context=1 for card-select). This caused INVALID at step 21
    of episode 89328390. All context matching now handles both int and string.
    """
    global _state

    try:
        # ── Turn 0: submit deck ──────────────────────────────────────────────
        step = obs_dict.get("step", 1) if obs_dict else 0
        if step == 0 or obs_dict is None:
            return _read_deck()

        select_data = obs_dict.get("select")
        if not select_data:
            return []

        options = select_data.get("option", [])
        if not options:
            return []

        max_count    = select_data.get("maxCount", 1)
        select_type  = select_data.get("type", 0)
        select_ctx   = select_data.get("context", "")

        curr = obs_dict.get("current") or {}
        turn = curr.get("turn", 0)
        me, opp = _get_players(obs_dict)

        # Reset per-turn supporter tracker on new turn
        if turn != _state["last_turn"]:
            _state["last_turn"] = turn
            _state["supporter_played_this_turn"] = False

        # ── YES/NO prompt (mulligan accept, etc.) ────────────────────────────
        if select_type == SELECT_YES_NO or _ctx_matches(select_ctx, 9, {"YESNO", "YES_NO", "ISFIRST"}):
            result = [0]  # always Yes / first option
            _debug_log(step, select_ctx, select_type, len(options), result)
            return result

        # ── ALL type=1 Card-select prompts ───────────────────────────────────
        # THE KEY FIX: previously fell through to [0] for integer contexts.
        if select_type == SELECT_CARD:
            result = _handle_card_select(select_data, me, opp)
            _debug_log(step, select_ctx, select_type, len(options), result)
            return result

        # ── MAIN phase (type=0) ──────────────────────────────────────────────
        if select_type == SELECT_MAIN:
            already_supporter = _state["supporter_played_this_turn"]
            want_retreat = _should_retreat_for_kilowattrel(options, me, opp, obs_dict)

            # 1. Ability first (Electric Streamer)
            ability_idx = _pick_ability(options, me)
            if ability_idx is not None:
                result = [ability_idx]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

            # 2. Evolve
            evolve_idx = _pick_evolve(options, me)
            if evolve_idx is not None:
                result = [evolve_idx]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

            # 3. Attach manual energy
            attach_idx = _pick_attach(options, me)
            if attach_idx is not None:
                result = [attach_idx]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

            # 4. Play cards (Boss, Poffin, Cheren, Switch, Catcher)
            play_idx = _pick_play_card(options, me, opp, already_supporter, obs_dict)
            if play_idx is not None:
                play_opts = _find_options_of_type(options, OPT_PLAY)
                for i, opt in play_opts:
                    if i == play_idx:
                        cid = _opt_card_id(opt)
                        if cid in (BOSS_ORDERS_ID, CHEREN_ID):
                            _state["supporter_played_this_turn"] = True
                result = [play_idx]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

            # 5. Retreat if needed
            if want_retreat:
                retreat_idx = _pick_retreat(options, me)
                if retreat_idx is not None:
                    result = [retreat_idx]
                    _debug_log(step, select_ctx, select_type, len(options), result)
                    return result

            # 6. Attack
            attack_idx = _pick_attack(options, me, opp)
            if attack_idx is not None:
                result = [attack_idx]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

            # 7. End turn
            end_opts = _find_options_of_type(options, OPT_END)
            if end_opts:
                result = [end_opts[0][0]]
                _debug_log(step, select_ctx, select_type, len(options), result)
                return result

        # ── §3.8.4 Final fallback — first legal option ───────────────────────
        result = [0]
        _debug_log(step, select_ctx, select_type, len(options), result)
        return result

    except Exception as e:
        # Absolute failsafe
        try:
            _debug_log(
                (obs_dict or {}).get('step', '?'),
                'EXCEPTION', str(e), 0, []
            )
            select_data = obs_dict.get("select", {}) if obs_dict else {}
            opts = select_data.get("option", []) if select_data else []
            if opts:
                return [0]
        except Exception:
            pass
        return []
