"""
Deep Replay Analyzer v3
Reads the 'visualize' blocks in cabt replays which have full game state (both players visible).
Identifies misplays by our agent turn-by-turn.
"""
import json, sys, os
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Card IDs
RIOLU=677; MEGA_LUCARIO=678; MAKUHITA=673; HARIYAMA=674
LUNATONE=675; SOLROCK=676; ENERGY=6
DUSK_BALL=1102; SWITCH=1123; PREMIUM_POWER=1141; FIGHTING_GONG=1142
POKE_PAD=1152; HERO_CAPE=1159; BOSS_ORDERS=1182; CARMINE=1192
LILLIE=1227; GRAVITY_MTN=1252

NAMES = {
    677:"Riolu",678:"Mega Lucario ex",673:"Makuhita",674:"Hariyama",
    675:"Lunatone",676:"Solrock",6:"F-Energy",1102:"Dusk Ball",
    1123:"Switch",1141:"Premium Power Pro",1142:"Fighting Gong",
    1152:"Poke Pad",1159:"Hero Cape",1182:"Boss Orders",
    1192:"Carmine",1227:"Lillie",1252:"Gravity Mtn",
}

def cn(i): return NAMES.get(i, f"#{i}")

def analyze(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    teams   = data.get("info", {}).get("TeamNames", ["P0","P1"])
    rewards = data.get("rewards", [None, None])

    # Identify our player
    our_idx = next((i for i,n in enumerate(teams) if "rahul" in n.lower()), 1)
    opp_idx = 1 - our_idx
    result  = "WIN" if rewards[our_idx] == 1 else "LOSS"

    misplays = []
    total_turns = 0

    steps = data.get("steps", [])
    for step in steps:
        # Try visualize blocks first (have full info)
        for agent_step in step:
            vis_list = agent_step.get("visualize", [])
            for vis in vis_list:
                cur = vis.get("current")
                sel = vis.get("select")
                if not cur or not sel:
                    continue

                context = sel.get("context", "")
                if context != "Main":
                    continue

                players = cur.get("players", [])
                if len(players) < 2:
                    continue

                # In visualize, yourIndex tells us whose turn it is
                your_idx = cur.get("yourIndex", 0)
                if your_idx != our_idx:
                    continue  # skip opponent's turns

                total_turns += 1
                me  = players[our_idx]
                opp = players[opp_idx]
                turn = cur.get("turn", 0)
                supp_played = cur.get("supporterPlayed", False)
                energy_attached = cur.get("energyAttached", False)
                selected = vis.get("selected")

                hand = [c["id"] for c in (me.get("hand") or []) if isinstance(c, dict)]
                bench = [p for p in (me.get("bench") or []) if isinstance(p, dict)]
                active_list = me.get("active") or []
                active = active_list[0] if active_list else {}
                active_id = active.get("id")
                active_hp = active.get("hp", 0)
                active_max_hp = active.get("maxHp", 1)
                active_energies = len(active.get("energyCards") or [])

                opp_active_list = opp.get("active") or []
                opp_active = opp_active_list[0] if opp_active_list else {}
                opp_bench = [p for p in (opp.get("bench") or []) if isinstance(p, dict)]
                opp_prizes = len(opp.get("prize") or [])
                my_prizes = len(me.get("prize") or [])

                # ---- MISPLAY CHECKS ----

                # 1. Energy in hand but not attached (turn > 2, active needs energy)
                energy_count = hand.count(ENERGY)
                if energy_count > 0 and not energy_attached and turn > 2:
                    if active_energies < 3:
                        misplays.append({
                            "turn": turn, "type": "ENERGY_NOT_ATTACHED",
                            "detail": f"Had {energy_count} energy in hand, active {cn(active_id)} "
                                      f"only has {active_energies} energy. Did not attach!"
                        })

                # 2. Boss's Orders: opponent has low-HP bench Pokémon
                if BOSS_ORDERS in hand and not supp_played:
                    for bp in opp_bench:
                        bp_hp  = bp.get("hp", 9999)
                        bp_max = bp.get("maxHp", 9999)
                        bp_id  = bp.get("id")
                        if bp_max > 0 and bp_hp / bp_max < 0.45:
                            misplays.append({
                                "turn": turn, "type": "BOSS_MISSED",
                                "detail": f"Had Boss Orders, opp bench {cn(bp_id)} at "
                                          f"{bp_hp}/{bp_max} HP ({100*bp_hp//bp_max}%). "
                                          f"Supporter not played!"
                            })
                            break

                # 3. Hero Cape not equipped to Lucario when Lucario is active and healthy
                # This is actually good — detect if we wait TOO long (lucario already at <50% HP)
                if HERO_CAPE in hand and active_id == MEGA_LUCARIO:
                    hp_frac = active_hp / max(active_max_hp, 1)
                    if hp_frac < 0.5:
                        misplays.append({
                            "turn": turn, "type": "HERO_CAPE_LATE",
                            "detail": f"Hero Cape in hand, Mega Lucario at {active_hp}/{active_max_hp} HP ({100*int(hp_frac)}%). Should have equipped earlier!"
                        })

                # 4. Bench filling: had searcher but bench < 3 and basics available in deck
                bench_basics = [p for p in bench if p["id"] in {RIOLU, MAKUHITA}]
                searcher_in_hand = DUSK_BALL in hand or POKE_PAD in hand
                if searcher_in_hand and len(bench) < 3 and turn > 1:
                    misplays.append({
                        "turn": turn, "type": "BENCH_UNDERFILLED",
                        "detail": f"Bench has {len(bench)} Pokémon (only {len(bench_basics)} basics). "
                                  f"Had {'Dusk Ball' if DUSK_BALL in hand else 'Poke Pad'} but bench not full."
                    })

                # 5. Attacking when Lucario is active with enough energy but turn ended without attack
                # Hard to detect precisely without knowing what options were taken
                # Instead: note when we have 2+ energy on Lucario but we're early in prizes
                if active_id == MEGA_LUCARIO and active_energies >= 2 and my_prizes > 0:
                    opts = sel.get("option") or []
                    attack_opts = [o for o in opts if o.get("type") == "Attack"]
                    if not attack_opts:
                        pass  # no attack available, fine
                    # If selected action was NOT an attack, flag it
                    elif selected is not None:
                        chosen_idx = selected[0] if selected else None
                        if chosen_idx is not None and chosen_idx < len(opts):
                            chosen_type = opts[chosen_idx].get("type", "")
                            if chosen_type != "Attack":
                                misplays.append({
                                    "turn": turn, "type": "ATTACK_SKIPPED",
                                    "detail": f"Mega Lucario had {active_energies} energies, "
                                              f"attack options available, but played {chosen_type} instead of attacking!"
                                })

                # 6. Carmine / Lillie when hand is already big
                if (CARMINE in hand or LILLIE in hand) and supp_played and len(hand) >= 5:
                    supp = "Carmine" if CARMINE in hand else "Lillie"
                    misplays.append({
                        "turn": turn, "type": "SUPPORTER_BIG_HAND",
                        "detail": f"{supp} played with {len(hand)} cards in hand — minimal draw value!"
                    })

                # 7. Premium Power Pro when Lucario hasn't attacked yet this turn
                if PREMIUM_POWER in hand and active_id == MEGA_LUCARIO:
                    # PPP gives extra attack if already attacked — should only be played post-attack
                    opts = sel.get("option") or []
                    # If we're playing PPP but haven't attacked (turn action count is low)
                    tac = cur.get("turnActionCount", 0)
                    if tac < 2:  # attacking usually happens as last action
                        pass  # can't easily infer without richer state

    return result, total_turns, misplays


def run_all():
    replay_files = [
        r"c:\Users\Rahul\Downloads\90325711.json",
        r"c:\Users\Rahul\Downloads\90326480.json",
        r"c:\Users\Rahul\Downloads\90327198.json",
        r"c:\Users\Rahul\Downloads\90327934.json",
        r"c:\Users\Rahul\Downloads\90328692.json",
        r"c:\Users\Rahul\Downloads\90329453.json",
        r"c:\Users\Rahul\Downloads\90330177.json",
        r"c:\Users\Rahul\Downloads\90330914.json",
        r"c:\Users\Rahul\Downloads\90331637.json",
        r"c:\Users\Rahul\Downloads\90332382.json",
        r"c:\Users\Rahul\Downloads\90333196.json",
        r"c:\Users\Rahul\Downloads\90334612.json",
        r"c:\Users\Rahul\Downloads\90333878.json",
        r"c:\Users\Rahul\Downloads\90335361.json",
        r"c:\Users\Rahul\Downloads\90336109.json",
        r"c:\Users\Rahul\Downloads\90336934.json",
        r"c:\Users\Rahul\Downloads\90337471.json",
        r"c:\Users\Rahul\Downloads\90338348.json",
        r"c:\Users\Rahul\Downloads\90339088.json",
        r"c:\Users\Rahul\Downloads\90339837.json",
        r"c:\Users\Rahul\Downloads\90340591.json",
        r"c:\Users\Rahul\Downloads\90341363.json",
        r"c:\Users\Rahul\Downloads\90353990.json",
    ]

    agg = defaultdict(list)
    wins, losses = 0, 0

    for fp in replay_files:
        eid = os.path.basename(fp)
        result, turns, misplays = analyze(fp)
        if result == "WIN": wins += 1
        else: losses += 1

        print(f"\n{'='*60}")
        print(f"Episode {eid}  |  {result}  |  Our turns: {turns}  |  Misplays found: {len(misplays)}")
        if misplays:
            for m in misplays:
                print(f"  [Turn {m['turn']:>3}] {m['type']}: {m['detail']}")
        for m in misplays:
            agg[m["type"]].append(m)

    print(f"\n{'='*60}")
    print(f"SUMMARY — Wins: {wins}  Losses: {losses}  Win rate: {100*wins//(wins+losses)}%")
    print(f"{'='*60}")
    for mtype, items in sorted(agg.items(), key=lambda x: -len(x[1])):
        print(f"  {mtype}: {len(items)} occurrences across all games")
        # Show top 3 examples
        for ex in items[:3]:
            print(f"    [T{ex['turn']}] {ex['detail'][:100]}")


if __name__ == "__main__":
    run_all()
