# 🏆 Roadmap: Climbing to 1200+ Elo

**Current best**: 265.3 (Phase 2 Pure NumPy) | **Target**: 1200+
**Entry deadline**: Aug 9, 2026 | **Final deadline**: Aug 16, 2026

---

## Rule #1 (NEVER BREAK)
> **Never promote a new agent to `main.py` unless its real ladder score beats the Phase 2 baseline of 265.3.**
> Every phase must be A/B tested on the actual Kaggle ladder before promotion.

---

## Phase 3 — PPO Foundation (🔄 IN PROGRESS)

**Goal**: Prove the True PPO + 645-dim state encoder architecture beats 265.3 on the real ladder.

**Status**: Training overnight — 200,000 episodes on RTX 3050.

### Steps (Morning after training completes):
- `[ ]` Check training logs. Win Rate should be 70%+. If < 60%, re-train.
- `[ ]` Run `scripts/build_submission.sh` to package `ppo_phase3.pth` + `data/`.
- `[ ]` Submit to Kaggle. Log result in `eval/ladder_log.md`.
- `[ ]` **Gate**: If score > 265.3, Phase 3 is promoted. Else, debug and re-train.

**Expected Elo**: ~350–500 (better state features + True PPO = better decisions)

---

## Phase 4 — Meta Deck Upgrade 🃏 ← NEXT

**Goal**: Replace the 4-Pokémon Snorlax deck (which loses to Bench Out immediately) with a
full competitive deck. Deck quality is the **single biggest lever** available.

### Final Deck Selection: "Iono's Bellibolt ex / Kilowattrel"

While Iron Boulder and Mega Gardevoir ex were strong candidates on paper, they both share a fatal flaw on the Kaggle ladder: **Crustle**.
If Crustle (which blocks all damage from ex/V/GX Pokémon) makes up ~50% of the Kaggle meta, pure ex decks like Mega Gardevoir ex will hard-cap at a 50% win rate. 

We are pivoting to the highest known Elo meta deck (836 Elo): **Iono's Bellibolt ex**.

#### Engine Analysis
- **Iono's Bellibolt ex** (ID 269): 280 HP. 
  - Ability: `Electric Streamer` (attach infinite {L} energy from hand)
  - Attack: `Thunderous Bolt` (230 damage)
- **Iono's Kilowattrel** (ID 271): 120 HP. **(Non-ex attacker)**
  - Ability: `Flashing Draw` (discard 1 {L} energy, draw to 6)
  - Attack: `Mach Bolt` (70 damage)
- **Iono's Tadbulb** (ID 268) & **Iono's Wattrel** (ID 270): 60 HP Basics.

#### Why this is mathematically perfect for the Agent
1. **The Crustle Answer**: Kilowattrel is a non-ex Stage 1. When the agent faces Crustle, it can pivot to Kilowattrel and win the match, bypassing the wall completely.
2. **Infinite Energy Attach**: `Electric Streamer` allows the agent to dump its hand of {L} energy onto the board instantly. This breaks the "1 energy per turn" rule, allowing 230 damage out of nowhere.
3. **Card Advantage**: `Flashing Draw` lets the agent refill its hand to 6 every turn. 

### The Pivot: RL to Hardcoded Spec

The provided `Iono_Bellibolt.pdf` outlines a **priority-ordered decision policy** for this specific deck. 
As noted in `AGENTS.md`, *"More sophisticated != better... a simpler heuristic scorer on the identical deck outperformed RL."* 

Instead of waiting for PPO to stumble into the optimal strategy, we are pivoting to implement a **bespoke Rule-Based Agent** exactly as defined in the PDF. This avoids the "hard-to-debug learned weights" failure mode and guarantees the agent will recognize the Crustle matchup and manage the Bellibolt cooldown perfectly.

### Final 60-Card List (Written to `deck.csv`)

```
# Pokemon (12)
4x Iono's Tadbulb (268)
4x Iono's Bellibolt ex (269)
2x Iono's Wattrel (270)
2x Iono's Kilowattrel (271)

# Trainers (28)
4x Buddy-Buddy Poffin (1086)
4x Ultra Ball (1121)
4x Boss's Orders (1182)
4x Cheren (1224)
4x Switch (1123)
4x Pokemon Catcher (1124)
2x Master Ball (1125)
2x Air Balloon (1174)

# Energy (20)
20x Basic {L} Lightning Energy (4)
```

### Steps:
- `[x]` Write `deck.csv` with Iono's Bellibolt list
- `[x]` Create `src/agent/rule_based_bellibolt.py` with full 7-step priority policy
- `[x]` Crustle ex-immunity detection branch (§3.3 step 1)
- `[x]` Bellibolt cooldown via engine signal (Thunderous Bolt absent = cooldown)
- `[x]` §3.8.4 `_legal_fallback` — triple-wrapped try/except
- `[x]` Point `main.py` to new agent
- `[x]` Build & submit to Kaggle (Phase 4 submission — awaiting ladder score)

**Expected Elo jump**: +500 to +600 (Guaranteed optimal execution of an 836 Elo deck)

---

## Phase 5 — RL + Heuristic Hybrid 🤖

> [!IMPORTANT]
> **Only begins if Phase 4 gets a real ladder score. Hard cutoff: Aug ~11.**
> If the hybrid is not clearly beating the rule-based baseline's ladder score by ~Aug 11,
> submit the Phase 4 rule-based agent as the final entry and stop.

**Goal**: Build on top of the rule-based agent, not replace it.

### Design: `src/agent/hybrid_agent.py`

1. **Log all Phase 4 ladder games as self-play training data** going forward.
   - The rule-based agent's action choices form a supervised "expert" signal.
   - This bootstraps PPO to play the Bellibolt deck correctly from day 1.

2. **Reward-shape against rule-based agent's action choices** early in training.
   - Add an auxiliary imitation loss: `λ * KL(policy || rule_based_action)`.
   - Anneal λ → 0 as training stabilises (after ~20k episodes).
   - Prevents the RL agent from "forgetting" the Crustle matchup branch.

3. **Keep rule-based engine as a runtime fallback inside the hybrid agent.**
   - If the policy's max softmax logit < confidence_threshold (0.6) → use rule-based decision.
   - This is NOT just a benchmark we retire. It's a permanent guardrail.
   - Guarantees the hybrid never plays worse than the rule-based baseline on any turn.

4. **Hard cutoff rule**:
   - If hybrid doesn't clearly beat Phase 4 ladder score by ~Aug 11 → revert to Phase 4.
   - Never promote the hybrid to `main.py` without a positive real-ladder delta.

**Implementation files**:
- `[ ]` `src/agent/hybrid_agent.py` — policy inference with rule-based fallback
- `[ ]` `src/train/train_ppo.py` — add imitation loss term (λ-annealed)
- `[ ]` `src/train/self_play.py` — log rule-based decisions as expert data

**Expected Elo jump vs Phase 4**: +100 to +300

---

## Phase 6 — Opponent Modeling 🔍

**Goal**: Use `src/agent/opponent_model.py` to track revealed/discarded opponent cards and feed a probabilistic hand estimate into the state encoder.

**How it works**:
- Maintain a running count of all cards the opponent has played/discarded (visible).
- Use Bayesian inference to estimate probability of each card type in opponent's unseen hand.
- Append a 15-dim probability vector to the state encoder output (645 → 660 dims).

**Impact**: Allows the agent to make better decisions when it matters most — e.g., not attaching energy when opponent likely has a Gust of Wind in hand.

**Expected Elo jump**: +100 to +200

---

## Timeline (Updated)

| Date | Phase | Action |
|---|---|---|
| **Aug 1 (done)** | Phase 4 | Rule-based Bellibolt agent written + `main.py` updated. |
| **Aug 1** | Phase 4 | Run `cabt_eval.py` locally. Submit to Kaggle. |
| **Aug 2** | Phase 4 | Log Phase 4 ladder result. Gate: > 303.6 to proceed. |
| **Aug 3–4** | Phase 5 | Begin hybrid RL training with imitation loss. |
| **Aug 5** | Phase 5 | Submit hybrid agent. Validate > Phase 4 ladder score. |
| **Aug 6–7** | Phase 5/6 | Tune hybrid. Optionally add opponent modeling. |
| **Aug 8** | Phase 5 | Final A/B decision: hybrid vs rule-based for entry. |
| **Aug 9** | **ENTRY DEADLINE** | Entry submission must be in by midnight. |
| **Aug 10–15** | Polish | Iterate, submit daily (max 5/day, only 2 active). |
| **~Aug 11** | **HARD CUTOFF** | If hybrid not beating Phase 4, revert to rule-based. |
| **Aug 16** | **FINAL DEADLINE** | Final submission. |



---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 3 score < 265.3 | Medium | Debug state encoder shape mismatch; re-train 50k more episodes |
| MCTS times out mid-match | High | Hard cap at 100 simulations; fallback to greedy policy |
| Meta deck hurts RL learning | Medium | Fall back to Phase 3 model + new deck (agent generalizes) |
| Local sim doesn't predict ladder | Confirmed | Always validate on real ladder before promoting |

---

## North Star Metric

> **Elo > 1200** = top-tier performance. Every phase must demonstrate a real ladder improvement before the next phase begins.

Currently at **265.3**. With all 3 upgrades (Deck + MCTS + Opponent Model), a target of **800–1200+** is realistic within the deadline.
