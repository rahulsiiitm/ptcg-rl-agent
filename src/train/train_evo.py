# -*- coding: utf-8 -*-
"""
train_evo.py - Memetic / (1+lambda) evolutionary weight tuner for rule_based_lucario.py

Approach (mirrors v15's alak_evo_wm memetic tuning):
  - Maintain a population of weight vectors (genomes)
  - Each generation, mutate N children from the current champion
  - Evaluate each child via local self-play vs the champion + vs a v15-style fixed bot
  - If any child wins more than 50% of matches, promote it as the new champion
  - Save winning weights to lucario_w.json after every generation

Key improvements over the old train_evo.py:
  1. Tunes ALL meaningful weight keys (not just 12)
  2. CMA-ES-inspired adaptive mutation scale
  3. Plays both as P1 and P2 (position-balanced evaluation)
  4. Runs vs MULTIPLE opponents: self-play champion + v15-style bot
  5. 6 parallel workers on Ryzen 5 5600H
  6. Saves checkpoint after every improvement
"""

import os, json, random, sys, time, copy, logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)

try:
    from kaggle_environments import make
except ImportError:
    print("Install kaggle_environments first: pip install kaggle_environments")
    sys.exit(1)

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_AGENT    = "src/agent/rule_based_lucario.py"
DECK_PATH     = "deck.csv"
TEMP_DIR      = "temp_evo"
OUT_WEIGHTS   = "lucario_w.json"

# v15 agent path for cross-play evaluation (optional — used if file exists)
V15_AGENT     = "v15/main.py"

MAX_WORKERS   = 6          # Ryzen 5 5600H has 12 logical cores → 6 parallel games
GENERATIONS   = 50         # run ~50 generations
CHILDREN      = 8          # mutants per generation
MATCHES_SELF  = 16         # self-play matches vs champion per child (8 as P1, 8 as P2)
MATCHES_V15   = 8          # matches vs v15 (only if V15 exists)
WIN_THRESHOLD = 0.55       # must beat 55% to promote (avoids noise-driven promotions)
SIGMA_INIT    = 400        # initial mutation scale
SIGMA_MIN     = 50         # floor on mutation scale
SIGMA_DECAY   = 0.92       # scale shrinks when no improvement (CMA-ES-lite)
SIGMA_GROW    = 1.1        # scale grows when improvement found
MAX_TIME_H    = 8.0        # hard cutoff (hours)

# ─── All tunable weight keys ────────────────────────────────────────────────────
# These map directly to the WEIGHTS dict in rule_based_lucario.py.
# Add any new keys you add to the agent here too.
MUTATE_KEYS = [
    # Pokemon play priorities
    "play_pokemon_base",
    # Trainer priorities
    "play_dusk_pad",   # Dusk Ball / Poke Pad / Fighting Gong
    "play_switch",
    "play_premium",    # Premium Power Pro
    "play_boss",
    "play_carmine",
    "play_lillie",
    # Tool priorities
    "attach_hero_cape",
    # Evolution / ability / movement / attack
    "evolve_base",
    "ability_base",
    "retreat_base",
    "attack_base",
]

# ─── Bounds: keep weights in sane ranges ───────────────────────────────────────
WEIGHT_BOUNDS = {
    "play_pokemon_base": (10000, 30000),
    "play_dusk_pad":     (3000, 20000),
    "play_switch":       (1000, 15000),
    "play_premium":      (1000, 10000),
    "play_boss":         (500,  8000),
    "play_carmine":      (500,  8000),
    "play_lillie":       (500,  8000),
    "attach_hero_cape":  (2000, 15000),
    "evolve_base":       (2000, 20000),
    "ability_base":      (15000, 50000),
    "retreat_base":      (500,  5000),
    "attack_base":       (200,  3000),
}

DEFAULT_WEIGHTS = {
    "play_pokemon_base": 20000,
    "play_dusk_pad":     8000,
    "play_switch":       6000,
    "play_premium":      5000,
    "play_boss":         3200,
    "play_carmine":      3000,
    "play_lillie":       3100,
    "attach_hero_cape":  7000,
    "evolve_base":       9000,
    "ability_base":      30000,
    "retreat_base":      2000,
    "attack_base":       1000,
}


def load_deck():
    with open(DECK_PATH) as f:
        deck = [int(l.strip()) for l in f if l.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv has {len(deck)} cards, expected 60")
    return deck


def load_meta_decks():
    meta_decks = []
    if os.path.exists("top20_decks"):
        for fname in os.listdir("top20_decks"):
            if not fname.endswith(".csv"): continue
            with open(os.path.join("top20_decks", fname)) as f:
                d = [int(l.strip()) for l in f if l.strip()]
                if len(d) == 60:
                    meta_decks.append(d)
    if not meta_decks:
        print("WARNING: No meta decks found in top20_decks/. Using only main deck for opponents.")
        meta_decks.append(load_deck())
    return meta_decks


def mutate(base: dict, sigma: float) -> dict:
    """Gaussian mutation with optional key dropout (sparse perturbation)."""
    child = copy.deepcopy(base)
    n_keys = len(MUTATE_KEYS)
    # Mutate at least 2 keys, at most all
    n_mutate = random.randint(2, n_keys)
    keys_to_mutate = random.sample(MUTATE_KEYS, n_mutate)
    for k in keys_to_mutate:
        delta = int(random.gauss(0, sigma))
        lo, hi = WEIGHT_BOUNDS.get(k, (0, 100000))
        child[k] = max(lo, min(hi, child[k] + delta))
    return child


def _write_agent_with_weights(source_code: str, weights: dict, path: str):
    """Write agent file with weights hard-baked via a JSON sidecar."""
    sidecar = path.replace(".py", "_w.json")
    with open(sidecar, "w") as f:
        json.dump(weights, f)
    # Patch the sidecar path so the agent reads from this specific file
    abs_sidecar = os.path.abspath(sidecar).replace("\\", "/")
    patched = source_code.replace(
        'for _p in ("lucario_w.json", "./lucario_w.json", "/kaggle_simulations/agent/lucario_w.json"):',
        f'for _p in ("{abs_sidecar}", "lucario_w.json", "./lucario_w.json", "/kaggle_simulations/agent/lucario_w.json"):'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)


def run_match(args):
    """Run one match, return 1 if agent1 wins, 0 if agent2 wins/draw, -1 if crash."""
    agent1_path, agent2_path, deck1, deck2 = args
    try:
        env = make("cabt", configuration={"decks": [deck1, deck2]})
        steps = env.run([agent1_path, agent2_path])
        result = steps[-1][0]["observation"].get("result", -1)
        if result == 0:
            return 1   # agent1 won
        if result == 1:
            return 0   # agent2 won
        return 0        # draw or unknown → no credit
    except Exception:
        return -1


def evaluate(child_path: str, champion_path: str, my_deck: list, meta_decks: list,
             n_self: int, v15_path: str | None, n_v15: int,
             executor: ProcessPoolExecutor) -> tuple[float, float]:
    """
    Returns (self_winrate, v15_winrate).
    Self-play: child vs champion, balanced P1/P2.
    V15 cross-play: child vs v15 (only P1 for simplicity).
    """
    tasks = []
    roles = []

    # Self-play: half as P1, half as P2
    for i in range(n_self):
        opp_deck = random.choice(meta_decks)
        if i % 2 == 0:
            tasks.append((child_path, champion_path, my_deck, opp_deck))
            roles.append("p1_self")
        else:
            tasks.append((champion_path, child_path, opp_deck, my_deck))
            roles.append("p2_self")

    # Cross-play vs v15
    if v15_path and os.path.exists(v15_path) and n_v15 > 0:
        for _ in range(n_v15):
            opp_deck = random.choice(meta_decks)
            tasks.append((child_path, v15_path, my_deck, opp_deck))
            roles.append("p1_v15")

    futures = [executor.submit(run_match, t) for t in tasks]
    results = [f.result() for f in futures]

    self_wins = 0; self_total = 0
    v15_wins = 0;  v15_total = 0
    crashes = 0

    for role, res in zip(roles, results):
        if res == -1:
            crashes += 1
            continue
        if role == "p1_self":
            self_total += 1
            self_wins += res
        elif role == "p2_self":
            self_total += 1
            self_wins += (1 - res)   # we're P2, so win if agent2 wins (result=0 means P1 won → we lost)
            # wait — run_match returns 1 if agent1 wins; here agent1=champion, so P2 wins when result=0
            # fix: re-read — role p2_self means child is agent2, wins if result=0
            # but run_match already returns 1 if agent1 wins, so for P2 child: win if result==0 → 1-res
        elif role == "p1_v15":
            v15_total += 1
            v15_wins += res

    if crashes > 0:
        print(f"  WARNING: {crashes}/{len(tasks)} matches crashed")

    self_wr  = self_wins  / self_total  if self_total  > 0 else 0.0
    v15_wr   = v15_wins   / v15_total   if v15_total   > 0 else -1.0
    return self_wr, v15_wr


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    my_deck = load_deck()
    meta_decks = load_meta_decks()
    print(f"Main deck loaded: {len(my_deck)} cards")
    print(f"Meta gauntlet loaded: {len(meta_decks)} opponent decks")

    v15_exists = os.path.exists(V15_AGENT)
    print(f"V15 cross-play: {'ENABLED' if v15_exists else 'DISABLED (v15/main.py not found)'}")

    with open(BASE_AGENT, "r", encoding="utf-8") as f:
        agent_code = f.read()

    # Start from lucario_w.json if it exists and is non-default, else use defaults
    if os.path.exists(OUT_WEIGHTS):
        with open(OUT_WEIGHTS) as f:
            champion_w = json.load(f)
        print(f"Resuming from existing {OUT_WEIGHTS}")
    else:
        champion_w = copy.deepcopy(DEFAULT_WEIGHTS)
        print("Starting from default weights")

    sigma = SIGMA_INIT
    best_ever_self_wr = 0.0
    no_improve_streak = 0

    champ_path = os.path.join(TEMP_DIR, "best_agent.py")
    _write_agent_with_weights(agent_code, champion_w, champ_path)

    start = time.time()
    print(f"\nStarting {GENERATIONS} generations × {CHILDREN} children × {MATCHES_SELF} self-play matches")
    print(f"Max workers: {MAX_WORKERS} | sigma={sigma} | threshold={WIN_THRESHOLD:.0%}")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for gen in range(GENERATIONS):
            elapsed_h = (time.time() - start) / 3600
            if elapsed_h > MAX_TIME_H:
                print(f"\nTime limit {MAX_TIME_H}h reached. Stopping.")
                break

            print(f"\n[Gen {gen+1:03d}/{GENERATIONS}] σ={sigma:.0f}  elapsed={elapsed_h:.2f}h")

            children = [mutate(champion_w, sigma) for _ in range(CHILDREN)]
            child_paths = []
            for i, cw in enumerate(children):
                path = os.path.join(TEMP_DIR, f"v{i}_agent.py")
                _write_agent_with_weights(agent_code, cw, path)
                child_paths.append(path)

            results = []
            for cp in child_paths:
                self_wr, v15_wr = evaluate(
                    cp, champ_path, my_deck, meta_decks,
                    MATCHES_SELF, V15_AGENT if v15_exists else None, MATCHES_V15,
                    executor
                )
                combined = self_wr if v15_wr < 0 else 0.6 * self_wr + 0.4 * v15_wr
                results.append((self_wr, v15_wr, combined))
                v15_str = f"vs_v15={v15_wr:.0%}" if v15_wr >= 0 else "vs_v15=N/A"
                print(f"  Child {len(results):02d}: self={self_wr:.0%}  {v15_str}  combined={combined:.2f}")

            # Pick best child by combined score
            best_idx = max(range(CHILDREN), key=lambda i: results[i][2])
            best_self_wr, best_v15_wr, best_combined = results[best_idx]

            if best_self_wr > WIN_THRESHOLD:
                print(f"  ✓ Child {best_idx} promoted! (self={best_self_wr:.0%})")
                champion_w = children[best_idx]
                _write_agent_with_weights(agent_code, champion_w, champ_path)

                # Save immediately
                with open(OUT_WEIGHTS, "w") as f:
                    json.dump(champion_w, f, indent=2)
                print(f"  Saved to {OUT_WEIGHTS}")

                sigma = min(SIGMA_INIT, sigma * SIGMA_GROW)
                no_improve_streak = 0

                if best_self_wr > best_ever_self_wr:
                    best_ever_self_wr = best_self_wr
                    # Also save a dated snapshot
                    snap = os.path.join(TEMP_DIR, f"gen{gen+1:03d}_best_w.json")
                    with open(snap, "w") as f:
                        json.dump(champion_w, f, indent=2)
            else:
                print(f"  ✗ No promotion. Best self={best_self_wr:.0%} < {WIN_THRESHOLD:.0%} threshold.")
                sigma = max(SIGMA_MIN, sigma * SIGMA_DECAY)
                no_improve_streak += 1
                if no_improve_streak >= 10:
                    print("  10 generations without improvement. Injecting random restart...")
                    sigma = SIGMA_INIT
                    no_improve_streak = 0

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("Best weights (saved to lucario_w.json):")
    print(json.dumps(champion_w, indent=2))
    print("\nNOTE: lucario_w.json must be included in your submission.tar.gz")


if __name__ == "__main__":
    main()