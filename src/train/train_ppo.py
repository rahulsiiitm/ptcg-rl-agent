"""
Phase 3 PPO training with multiprocessing parallel environments.

Architecture:
  - N worker processes each own one FastPTCGEnv
  - Main process runs batched GPU forward pass and sends actions back via Pipes
  - Self-play: opponent pool sampled from models/pool/
  - Saves checkpoint every CHECKPOINT_EVERY episodes
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process
import time

import os
import sys
# Add project root to sys.path dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import get_action_mask, MAX_ACTION_SPACE

# ─── Hyperparameters ──────────────────────────────────────────────────────────
STATE_DIM       = ObservationEncoder.STATE_DIM   # 167
ACTION_DIM      = MAX_ACTION_SPACE               # 150
HIDDEN_DIM      = 256
N_WORKERS       = 8          # parallel envs (tune to your CPU core count)
EPISODES        = 100_000    # overnight training total episodes
MAX_STEPS       = 300        # max steps per episode
GAMMA           = 0.99
CLIP_RATIO      = 0.2
LR              = 3e-4
ENTROPY_COEF    = 0.01       # encourage exploration
CHECKPOINT_EVERY = 5_000     # save to pool every N episodes
POOL_DIR        = "models/pool"

# Load deck directly from deck.csv to ensure inference exactly matches training
def _load_deck_from_csv(path="deck.csv"):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]
SNORLAX_DECK = _load_deck_from_csv()

# ─── Model ────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


# ─── Worker process ───────────────────────────────────────────────────────────

def env_worker(worker_id: int, conn: mp.connection.Connection, rl_deck: list, opp_deck: list):
    """
    Each worker owns one environment. Protocol:
      - Receives action list from main process
      - Sends back (obs, reward, done)
      - Receives 'reset' to start a new episode
      - Receives 'close' to terminate
    """
    sys.path.append(os.getcwd())
    from src.env.fast_sim import FastPTCGEnv
    from src.train.self_play import SelfPlayPool

    pool = SelfPlayPool()
    env = FastPTCGEnv(rl_deck=rl_deck, opp_deck=opp_deck)
    env.set_opponent_agent(pool.sample_opponent())
    obs, _ = env.reset()
    conn.send(('obs', obs))

    while True:
        msg = conn.recv()
        if msg == 'reset':
            env.set_opponent_agent(pool.sample_opponent())
            obs, _ = env.reset()
            conn.send(('obs', obs))
        elif msg == 'close':
            break
        else:
            # msg is the action list
            action = msg
            obs, reward, done, _, _ = env.step(action)
            conn.send(('step', obs, reward, done))



# ─── Export for Kaggle submission ─────────────────────────────────────────────

def export_weights(model: ActorCritic, path: str = "models/ppo_weights.npz"):
    """Export actor weights to NumPy for Kaggle (no torch at inference)."""
    w = {}
    # shared
    w['s1_w'] = model.shared[0].weight.detach().cpu().numpy()
    w['s1_b'] = model.shared[0].bias.detach().cpu().numpy()
    w['s2_w'] = model.shared[2].weight.detach().cpu().numpy()
    w['s2_b'] = model.shared[2].bias.detach().cpu().numpy()
    # actor head
    w['a_w'] = model.actor.weight.detach().cpu().numpy()
    w['a_b'] = model.actor.bias.detach().cpu().numpy()
    np.savez(path, **w)
    print(f"  Exported NumPy weights to {path}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  |  Workers: {N_WORKERS}  |  Episodes: {EPISODES}")

    os.makedirs(POOL_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    encoder = ObservationEncoder()
    model   = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Spawn worker processes
    workers, parent_conns, child_conns = [], [], []
    for i in range(N_WORKERS):
        parent_conn, child_conn = Pipe()
        p = Process(
            target=env_worker,
            args=(i, child_conn, SNORLAX_DECK, SNORLAX_DECK),
            daemon=True
        )
        p.start()
        workers.append(p)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # Collect initial observations
    obs_list = []
    for conn in parent_conns:
        tag, obs = conn.recv()
        obs_list.append(obs)

    ep = 0
    total_wins = 0
    t0 = time.time()

    try:
        while ep < EPISODES:
            # ── Rollout: one step across all N_WORKERS ──
            states = np.stack([encoder.encode(o) for o in obs_list])   # (N, STATE_DIM)
            state_t = torch.tensor(states, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits, values = model(state_t)

            actions, log_probs = [], []
            for i, obs in enumerate(obs_list):
                mask = get_action_mask(obs)
                mask_t = torch.from_numpy(mask).to(device)
                masked = logits[i].clone()
                masked[~mask_t] = -1e9
                probs = torch.softmax(masked, dim=-1)

                if probs.sum() < 1e-8 or torch.isnan(probs).any():
                    action = [0]
                    log_prob = torch.tensor(0.0, device=device)
                else:
                    select_data = obs.get("select", {})
                    max_count = select_data.get("maxCount", 1)
                    num_valid = mask_t.sum().item()
                    max_count = min(max_count, num_valid)
                    
                    if max_count <= 1:
                        dist = torch.distributions.Categorical(probs)
                        idx  = dist.sample()
                        action = [idx.item()]
                        log_prob = dist.log_prob(idx)
                    else:
                        # Sample without replacement for max_count > 1
                        action = []
                        log_prob_sum = torch.tensor(0.0, device=device)
                        curr_probs = probs.clone()
                        for _ in range(max_count):
                            dist = torch.distributions.Categorical(curr_probs)
                            idx = dist.sample()
                            action.append(idx.item())
                            log_prob_sum += dist.log_prob(idx)
                            # Mask out the chosen action
                            curr_probs[idx] = 0.0
                            if curr_probs.sum() > 0:
                                curr_probs = curr_probs / curr_probs.sum()
                            else:
                                break
                        log_prob = log_prob_sum

                actions.append(action)
                log_probs.append(log_prob)

            # Send actions to workers
            for conn, action in zip(parent_conns, actions):
                conn.send(action)

            # Collect results
            new_obs_list, rewards, dones = [], [], []
            for conn in parent_conns:
                tag, obs, reward, done = conn.recv()
                new_obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)

            # ── PPO update ──
            rewards_t   = torch.tensor(rewards,   dtype=torch.float32).to(device)
            dones_t     = torch.tensor(dones,     dtype=torch.float32).to(device)
            log_probs_t = torch.stack(log_probs)

            # Next values for TD target
            next_states = np.stack([encoder.encode(o) for o in new_obs_list])
            next_t = torch.tensor(next_states, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, next_values = model(next_t)
                next_values = next_values.squeeze(-1)

            td_targets = rewards_t + GAMMA * next_values * (1.0 - dones_t)
            advantages = (td_targets - values.squeeze(-1)).detach()

            actor_loss  = -(log_probs_t * advantages).mean()
            critic_loss = advantages.pow(2).mean()
            # Re-compute entropy for bonus
            state_t2 = torch.tensor(states, dtype=torch.float32).to(device)
            logits2, _ = model(state_t2)
            probs2 = torch.softmax(logits2, dim=-1)
            entropy = -(probs2 * (probs2 + 1e-8).log()).sum(-1).mean()
            loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # ── Episode bookkeeping ──
            for i, done in enumerate(dones):
                if done:
                    ep += 1
                    result = new_obs_list[i].get('current', {})
                    if result:
                        r = result.get('result', -1)
                        if r == 0:
                            total_wins += 1

                    if ep % 500 == 0:
                        win_rate = total_wins / ep
                        elapsed = time.time() - t0
                        print(f"  Ep {ep:>6}/{EPISODES}  WinRate={win_rate:.1%}  Loss={loss.item():.4f}  [{elapsed:.0f}s]")

                    if ep % CHECKPOINT_EVERY == 0:
                        ckpt = os.path.join(POOL_DIR, f"checkpoint_{ep}.pth")
                        torch.save(model.state_dict(), ckpt)
                        print(f"  Checkpoint saved to {ckpt}")

                    # Reset this worker
                    parent_conns[i].send('reset')
                    tag, obs = parent_conns[i].recv()
                    new_obs_list[i] = obs

            obs_list = new_obs_list

    finally:
        # Shutdown workers
        for conn in parent_conns:
            try: conn.send('close')
            except: pass
        for p in workers:
            p.join(timeout=2)

    # Save final model
    torch.save(model.state_dict(), "models/ppo_phase3.pth")
    print("Saved to models/ppo_phase3.pth")

    # Export NumPy weights for Kaggle
    export_weights(model, "models/ppo_weights.npz")

    win_rate = total_wins / max(ep, 1)
    print(f"\nTraining complete. Final win rate vs rule-based: {win_rate:.1%} over {ep} episodes.")
    return model


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
