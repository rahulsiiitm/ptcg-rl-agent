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
import glob

# Add project root to sys.path dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.state_encoder import ObservationEncoder
from src.agent.action_mask import get_action_mask, MAX_ACTION_SPACE
from src.agent.rule_based_lopunny import rule_based_agent as expert_agent

# ─── Hyperparameters ──────────────────────────────────────────────────────────
STATE_DIM       = ObservationEncoder.STATE_DIM   # 167
ACTION_DIM      = MAX_ACTION_SPACE               # 150
HIDDEN_DIM      = 256
N_WORKERS       = 9          # parallel envs (tune to your CPU core count)
EPISODES        = 100_000
MAX_STEPS       = 300        # max steps per episode
GAMMA           = 0.99
CLIP_RATIO      = 0.2
LR              = 3e-4
ENTROPY_COEF    = 0.01       # encourage exploration
CHECKPOINT_EVERY = 5_000     # save to pool every N episodes
POOL_DIR        = "models/pool"

IMITATION_LAMBDA = 0.8
IMITATION_EPISODES = 20_000

# Load deck directly from deck.csv to ensure inference exactly matches training
def _load_deck_from_csv(path="deck.csv"):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]

DECK_DIR = "decks"
SNORLAX_DECK = _load_deck_from_csv(os.path.join(DECK_DIR, "lopunny_froslass_ids.csv"))

ALL_DECKS = []
for p in glob.glob(os.path.join(DECK_DIR, "meta_*.csv")):
    try:
        ALL_DECKS.append(_load_deck_from_csv(p))
    except Exception as e:
        print(f"Skipping {p}: {e}")

if SNORLAX_DECK not in ALL_DECKS:
    ALL_DECKS.append(SNORLAX_DECK)

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
        self.q_critic = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h), self.q_critic(h)


# ─── Worker process ───────────────────────────────────────────────────────────

def env_worker(_, conn: mp.connection.Connection, all_decks: list):
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
    import random

    pool = SelfPlayPool()
    
    # Lazy import to avoid circular dependencies in workers
    from src.agent.rule_based_generic import rule_based_generic_agent
    
    initial_opp_deck = random.choice(all_decks)
    env = FastPTCGEnv(rl_deck=SNORLAX_DECK, opp_deck=initial_opp_deck)
    
    if initial_opp_deck == SNORLAX_DECK:
        env.set_opponent_agent(pool.sample_opponent())
    else:
        env.set_opponent_agent(rule_based_generic_agent)
        
    obs, _ = env.reset()
    conn.send(('obs', obs))

    while True:
        msg = conn.recv()
        if isinstance(msg, tuple) and msg[0] == 'reset':
            env.set_rl_deck(msg[1])
            env.set_opponent_deck(msg[2])
            
            if msg[2] == SNORLAX_DECK:
                env.set_opponent_agent(pool.sample_opponent())
            else:
                env.set_opponent_agent(rule_based_generic_agent)
                
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

    encoders = [ObservationEncoder() for _ in range(N_WORKERS)]
    model   = ActorCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN_DIM).to(device)
    
    # Resume from latest checkpoint or BC pre-training
    START_EP = 0
    import glob
    checkpoints = glob.glob("models/pool/checkpoint_*.pth")
    if checkpoints:
        # Sort by episode number
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]), reverse=True)
        latest_ckpt = checkpoints[0]
        START_EP = int(os.path.basename(latest_ckpt).split('_')[1].split('.')[0])
        model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=True), strict=False)
        print(f"Resuming from {latest_ckpt} checkpoint at episode {START_EP}!")
    elif os.path.exists("models/ppo_latest.pth"):
        model.load_state_dict(torch.load("models/ppo_latest.pth", map_location=device, weights_only=True), strict=False)
        print("Resuming from existing models/ppo_latest.pth checkpoint!")
    elif os.path.exists("models/bc_model.pth"):
        model.load_state_dict(torch.load("models/bc_model.pth", map_location=device, weights_only=True), strict=False)
        print("Resuming from existing models/bc_model.pth checkpoint!")
        
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Spawn worker processes
    workers, parent_conns, child_conns = [], [], []
    for i in range(N_WORKERS):
        parent_conn, child_conn = Pipe()
        p = Process(
            target=env_worker,
            args=(i, child_conn, ALL_DECKS),
            daemon=True
        )
        p.start()
        workers.append(p)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # PPO Hyperparams
    STEPS_PER_ROLLOUT = 128
    PPO_EPOCHS = 4
    MINIBATCH_SIZE = 256
    GAE_LAMBDA = 0.95
    
    # Collect initial observations
    obs_list = []
    for conn in parent_conns:
        tag, obs = conn.recv()
        obs_list.append(obs)

    ep = START_EP
    total_wins = 0
    t0 = time.time()
    
    deck_stats = {idx: {'wins': 0, 'games': 0} for idx in range(len(ALL_DECKS))}
    current_opp_deck_idx = [np.random.randint(len(ALL_DECKS)) for _ in range(N_WORKERS)]

    try:
        while ep < EPISODES:
            # ── Rollout Collection ──
            batch_states, batch_actions, batch_log_probs = [], [], []
            batch_rewards, batch_dones, batch_values, batch_masks = [], [], [], []
            batch_expert_actions = []
            
            for step in range(STEPS_PER_ROLLOUT):
                states = np.stack([encoders[i].encode(o) for i, o in enumerate(obs_list)])   # (N, STATE_DIM)
                state_t = torch.tensor(states, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    logits, values, q_values = model(state_t)
                    
                actions, log_probs = [], []
                masks = []
                expert_actions = []
                for i, obs in enumerate(obs_list):
                    expert_a = expert_agent(obs)
                    expert_actions.append(expert_a[0] if expert_a else 0)
                    
                    mask = get_action_mask(obs)
                    masks.append(mask)
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
                            # Sample without replacement
                            action = []
                            log_prob_sum = torch.tensor(0.0, device=device)
                            curr_probs = probs.clone()
                            for _ in range(max_count):
                                dist = torch.distributions.Categorical(curr_probs)
                                idx = dist.sample()
                                action.append(idx.item())
                                log_prob_sum += dist.log_prob(idx)
                                curr_probs[idx] = 0.0
                                if curr_probs.sum() > 0:
                                    curr_probs = curr_probs / curr_probs.sum()
                                else:
                                    break
                            log_prob = log_prob_sum

                    actions.append(action)
                    log_probs.append(log_prob)

                # Send actions
                for conn, action in zip(parent_conns, actions):
                    conn.send(action)

                # Collect results
                new_obs_list, rewards, dones = [], [], []
                for i, conn in enumerate(parent_conns):
                    tag, obs, reward, done = conn.recv()
                    new_obs_list.append(obs)
                    rewards.append(reward)
                    dones.append(done)
                    if done:
                        encoders[i].reset()
                
                # Append to buffers
                batch_states.append(state_t)
                batch_actions.append(actions)
                batch_log_probs.append(torch.stack(log_probs))
                batch_rewards.append(torch.tensor(rewards, dtype=torch.float32).to(device))
                batch_dones.append(torch.tensor(dones, dtype=torch.float32).to(device))
                batch_values.append(values.squeeze(-1))
                batch_masks.append(torch.tensor(np.stack(masks), dtype=torch.bool).to(device))
                batch_expert_actions.append(torch.tensor(expert_actions, dtype=torch.long).to(device))
                
                obs_list = new_obs_list
                
                # Bookkeeping
                for i, done in enumerate(dones):
                    if done:
                        ep += 1
                        result = obs_list[i].get('current', {})
                        r = result.get('result', -1) if result else -1
                        
                        deck_idx = current_opp_deck_idx[i]
                        deck_stats[deck_idx]['games'] += 1
                        if r == 0:
                            total_wins += 1
                            deck_stats[deck_idx]['wins'] += 1
                        
                        # 50% Mirror Match, 50% Meta Match
                        if np.random.rand() < 0.5:
                            # Mirror Match
                            parent_conns[i].send(('reset', SNORLAX_DECK, SNORLAX_DECK))
                        else:
                            # Meta Match
                            # Prioritized sampling: lower win rate = higher probability
                            probs = []
                            for d_idx in range(len(ALL_DECKS)):
                                st = deck_stats[d_idx]
                                wr = st['wins'] / max(1, st['games'])
                                probs.append(1.0 - wr + 0.1) # Add 0.1 for exploration
                            probs = np.array(probs)
                            probs /= probs.sum()
                            
                            sampled_opp_idx = np.random.choice(len(ALL_DECKS), p=probs)
                            current_opp_deck_idx[i] = sampled_opp_idx
                            
                            # Tell worker to reset with specific decks
                            parent_conns[i].send(('reset', SNORLAX_DECK, ALL_DECKS[sampled_opp_idx]))
                            
                        tag, new_obs = parent_conns[i].recv()
                        obs_list[i] = new_obs

                        if ep % 50 == 0:
                            win_rate = total_wins / max(1, ep - START_EP)
                            dt = time.time() - t0
                            print(f"Ep: {ep} | WR: {win_rate:.2f} | Time: {dt:.1f}s", flush=True)
                        
                        if ep % CHECKPOINT_EVERY == 0:
                            ckpt_path = os.path.join(POOL_DIR, f"checkpoint_{ep}.pth")
                            torch.save(model.state_dict(), ckpt_path)
                            print(f"Saved checkpoint to {ckpt_path}", flush=True)

            # ── GAE Advantage Computation ──
            # Get next value for the very last step
            states = np.stack([encoders[i].encode(o) for i, o in enumerate(obs_list)])
            next_t = torch.tensor(states, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, next_values, _ = model(next_t)
                next_values = next_values.squeeze(-1)
            
            advantages = torch.zeros_like(batch_rewards[0]).to(device)
            returns = []
            
            for t in reversed(range(STEPS_PER_ROLLOUT)):
                if t == STEPS_PER_ROLLOUT - 1:
                    next_non_terminal = 1.0 - batch_dones[t]
                    next_val = next_values
                else:
                    next_non_terminal = 1.0 - batch_dones[t]
                    next_val = batch_values[t + 1]
                
                delta = batch_rewards[t] + GAMMA * next_val * next_non_terminal - batch_values[t]
                advantages = delta + GAMMA * GAE_LAMBDA * next_non_terminal * advantages
                returns.insert(0, advantages + batch_values[t])
                
            # Flatten rollout tensors
            # shapes: (STEPS, WORKERS, ...) -> (STEPS * WORKERS, ...)
            b_states = torch.cat(batch_states)
            b_log_probs = torch.cat(batch_log_probs)
            b_returns = torch.cat(returns)
            b_advantages = b_returns - torch.cat(batch_values)
            b_masks = torch.cat(batch_masks)
            b_expert_actions = torch.cat(batch_expert_actions)
            
            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # Action lists need to be flattened carefully since they are lists of lists of ints
            flat_actions = []
            for t_actions in batch_actions:
                for w_action in t_actions:
                    flat_actions.append(w_action)
            
            # ── PPO Mini-batch Updates ──
            dataset_size = len(b_states)
            indices = np.arange(dataset_size)
            
            c2_imitation = max(0.0, IMITATION_LAMBDA * (1.0 - ep / IMITATION_EPISODES))
            
            for _ in range(PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_idx = indices[start:end]
                    
                    mb_states = b_states[mb_idx]
                    mb_masks = b_masks[mb_idx]
                    mb_log_probs_old = b_log_probs[mb_idx]
                    mb_returns = b_returns[mb_idx]
                    mb_advantages = b_advantages[mb_idx]
                    mb_expert_actions = b_expert_actions[mb_idx]
                    
                    logits, values, q_values = model(mb_states)
                    values = values.squeeze(-1)
                    
                    # Recompute log probs for actions
                    mb_log_probs = []
                    for k, idx in enumerate(mb_idx):
                        mask_t = mb_masks[k]
                        action_list = flat_actions[idx]
                        
                        if len(action_list) == 1:
                            masked = torch.where(mask_t, logits[k], torch.tensor(-1e9, device=device))
                            dist = torch.distributions.Categorical(logits=masked)
                            idx_t = torch.tensor(action_list[0], device=device)
                            mb_log_probs.append(dist.log_prob(idx_t))
                        else:
                            log_prob_sum = torch.tensor(0.0, device=device)
                            curr_mask = mask_t.clone()
                            for a in action_list:
                                masked = torch.where(curr_mask, logits[k], torch.tensor(-1e9, device=device))
                                dist = torch.distributions.Categorical(logits=masked)
                                idx_t = torch.tensor(a, device=device)
                                log_prob_sum = log_prob_sum + dist.log_prob(idx_t)
                                curr_mask = curr_mask.clone()
                                curr_mask[a] = False
                            mb_log_probs.append(log_prob_sum)
                            
                    mb_log_probs = torch.stack(mb_log_probs)
                    
                    # Policy Loss (Clipped Surrogate Objective)
                    # Clamp log_ratio to prevent exp() from overflowing and causing NaN gradients
                    log_ratio = torch.clamp(mb_log_probs - mb_log_probs_old, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value Loss
                    critic_loss = 0.5 * (mb_returns - values).pow(2).mean()
                    
                    # Entropy Bonus
                    masked_logits = torch.where(mb_masks, logits, torch.tensor(-1e9, device=device))
                    probs = torch.softmax(masked_logits, dim=-1)
                    entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
                    
                    # Imitation Loss
                    imitation_loss = nn.CrossEntropyLoss()(logits, mb_expert_actions)
                    
                    # Auxiliary Q-Value Loss
                    # We train q_values to predict the advantage + value (which is mb_returns)
                    # For the actions actually taken, their Q-value should match the return.
                    mb_primary_actions = torch.tensor([flat_actions[idx][0] for idx in mb_idx], dtype=torch.long, device=device)
                    q_loss = nn.MSELoss()(q_values.gather(1, mb_primary_actions.unsqueeze(1)).squeeze(-1), mb_returns)
                    
                    loss = actor_loss + critic_loss - ENTROPY_COEF * entropy + c2_imitation * imitation_loss + 0.5 * q_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        for conn in parent_conns:
            conn.send('close')
        for p in workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

        torch.save(model.state_dict(), "models/ppo_latest.pth")
        print("Saved models/ppo_latest.pth")
        
        try:
            export_weights(model, "models/ppo_weights.npz")
        except:
            pass

if __name__ == "__main__":
    mp.set_start_method("spawn")
    train()
