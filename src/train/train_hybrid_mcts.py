import os
import sys
import glob
import math
import random
import torch
import collections

sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0] if glob.glob('/kaggle/input/**/cg-lib', recursive=True) else os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cg.game import battle_start, battle_finish, battle_select
from src.agent.rule_based_lucario import agent as rule_based_agent
from src.agent.hybrid_lucario import (
    MyModel, LearnSample, LearnInput, mcts_agent, random_agent, 
    get_encoder_input, get_decoder_input, SparseVector
)
import src.agent.hybrid_lucario as hybrid_module

# RTX 3050 can handle more MCTS simulations per decision during training.
# Higher = better training data quality. Lower = faster epoch wall-clock time.
# 20 is a good balance for a 4 GB VRAM card.
hybrid_module.SEARCH_COUNT = 20

# Load the actual Lucario deck
deck_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deck.csv")
with open(deck_path, "r", encoding="utf-8") as f:
    lucario_deck = [int(line) for line in f.read().splitlines() if line.strip()]

# Load opponent decks
opponent_decks = []
decks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "top20_decks")
if os.path.exists(decks_dir):
    for fn in os.listdir(decks_dir):
        if fn.endswith('.csv'):
            with open(os.path.join(decks_dir, fn), "r", encoding="utf-8") as f:
                deck = [int(line) for line in f.read().splitlines() if line.strip()][:60]
                if len(deck) == 60:
                    opponent_decks.append(deck)

if not opponent_decks:
    opponent_decks = [lucario_deck]

def get_random_opponent_deck():
    return random.choice(opponent_decks)

def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if(current >= count):
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1

# ===== RTX 3050 Laptop GPU Tuning =====
# Ampere architecture (sm_86), 4 GB VRAM, 2048 CUDA cores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    # cuDNN auto-tuner: finds the best conv algorithm for your fixed input sizes
    torch.backends.cudnn.benchmark = True
    # TF32: Ampere GPUs can do 19-bit matmuls via Tensor Cores (speed vs full FP32)
    # This is a huge free speedup on RTX 30xx with no accuracy loss for RL
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[GPU] Using: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
else:
    print("[GPU] No CUDA GPU detected — running on CPU. Install CUDA PyTorch!")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")

# Model — MyModel(encoder_size, n_heads, hidden, enc_layers, dec_layers)
# Tuned for 4 GB VRAM: encoder_size=256 fits comfortably
model = MyModel(128, 2, 256, 1, 1)
model = model.to(device)

# --- Resume from checkpoint if it exists ---
if os.path.exists("model_latest.pth"):
    print("[INFO] Found model_latest.pth! Resuming training from previous checkpoint.")
    model.load_state_dict(torch.load("model_latest.pth", map_location=device))
else:
    print("[INFO] No checkpoint found. Starting training with random weights.")

# torch.compile gives ~15-30% speedup on Ampere for free (PyTorch 2.0+)
if device.type == "cuda":
    # Triton (required for torch.compile inductor backend) is not supported natively on Windows
    if sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[GPU] torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"[GPU] torch.compile failed: {e}")
    else:
        print("[GPU] torch.compile disabled (Windows does not support Triton natively)")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=(device.type == "cuda"))

loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function

os.makedirs("out", exist_ok=True)

replay_buffer = collections.deque(maxlen=50000)
# GradScaler for AMP: PyTorch 2.5 uses GradScaler("cuda") syntax
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
best_win_rate = -1.0
epoch = 0

# RTX 3050: 4 GB VRAM — BATCH_SIZE=256 fits well with FP16/AMP
# If you get OOM errors, reduce to 128
BATCH_SIZE = 256

# ---- Load pretrained replay data (Imitation Learning seed) ----
# Run `python scripts/parse_replays_to_training.py` first to generate this file.
_pretrain_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pretrain.pt")
if os.path.exists(_pretrain_path):
    _pretrain_samples = torch.load(_pretrain_path, weights_only=False)
    replay_buffer.extend(_pretrain_samples)
    print(f"[PRETRAIN] Loaded {len(_pretrain_samples)} samples from {_pretrain_path} into replay buffer.")
else:
    print(f"[PRETRAIN] No pretrain.pt found at {_pretrain_path}. Starting from scratch.")
    print("  -> Run: python scripts/parse_replays_to_training.py")

# The main training loop.
while True:
    epoch += 1
    print(f"\n=== Epoch {epoch} ===")
    torch.save(model.state_dict(), "model_latest.pth")

    sample_list:list[LearnSample] = []  # List of training data samples.

    

    model.eval()

    with torch.inference_mode():

        # Evaluation

        results = [0, 0, 0]



        for i in progress(50, "Evaluating... "):
            op_deck = get_random_opponent_deck()
            obs, start_data = battle_start(lucario_deck, op_deck)

            if start_data.errorPlayer >= 0:

                error = "Deck error."

                if start_data.errorType == 1:

                    error = "The deck contains invalid card ID."

                elif start_data.errorType == 2:

                    error = "You can include up to four cards with the same name in the deck, excluding basic Energy cards."

                elif start_data.errorType == 3:

                    error = "There are no Basic Pokémon in the deck."

                elif start_data.errorType == 4:

                    error = "You can include only one Ace Spec card in the deck."

                raise ValueError(error)

            your_index = i % 2

            while True:

                # Break the loop if the game has ended.

                if obs["current"]["result"] >= 0:

                    break



                if obs["current"]["yourIndex"] == your_index:

                    selected, _ = mcts_agent(obs, lucario_deck, model)

                else:
                    # Use rule-based agent as the opponent for a stronger eval baseline
                    selected = rule_based_agent(obs)

                obs = battle_select(selected)

            

            battle_finish()  # Finalize the game.



            if obs["current"]["result"] == 2:  # Draw

                results[2] += 1

            elif obs["current"]["result"] == your_index:  # Win

                results[0] += 1

            else: # Lose

                results[1] += 1

        win_rate = 100 * results[0] / max(1, (results[0] + results[1]))
        print("Evaluation win rate " + str(win_rate) + "%", flush=True)
        if win_rate >= best_win_rate:
            best_win_rate = win_rate
            torch.save(model.state_dict(), "model_best.pth")
            print("New best model saved!")



        # Self Play

        for _ in progress(100, "Training Data Collecting... "):
            op_deck = get_random_opponent_deck()
            # Randomize who goes first / which deck is player 0
            if random.random() < 0.5:
                p0_deck, p1_deck = lucario_deck, op_deck
            else:
                p0_deck, p1_deck = op_deck, lucario_deck
                
            obs, _ = battle_start(p0_deck, p1_deck)

            samples:list[list[LearnSample]] = [[], []]  # [Player0 samples, Player1 samples]

            while True:

                if obs["current"]["result"] >= 0:

                    break



                # The MCTS agent generates an action and a training sample.
                current_deck = p0_deck if obs["current"]["yourIndex"] == 0 else p1_deck
                selected, sample = mcts_agent(obs, current_deck, model)

                samples[obs["current"]["yourIndex"]].append(sample)

                obs = battle_select(selected)

            

            battle_finish()  # Finalize the game.



            # Calculate the training labels and add them to the training data list.

            for i in range(2):

                LAMBDA = 0.9

                # The final value is 1.0 for a win and -1.0 for a loss.

                value = 1.0 if i == obs["current"]["result"] else -1.0



                # Iterate backwards from the end of the game to calculate values.

                for sample in reversed(samples[i]):

                    label = (value + sample.value) * 0.5

                    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)

                    sample.value = label

                    sample_list.append(sample)



    # Train on the training data collected through self-play.
    print("Training Start.")
    model.train()
    
    # Append new samples to the replay buffer
    replay_buffer.extend(sample_list)
    
    # We will sample from the entire replay buffer
    train_data = list(replay_buffer)
    random.shuffle(train_data)

    BATCH_SIZE = 128
    
    # Limit to 500 batches per epoch to keep training fast
    batch_count = min(len(train_data) // BATCH_SIZE, 500)
    
    if batch_count == 0:
        print("Not enough data to train yet.")
        continue

    for i in range(batch_count):

        # Prepare a batch of data.

        input_enc = LearnInput()

        input_dec = LearnInput()

        mask = []

        label_enc = []

        label_dec = []

        start = BATCH_SIZE * i

        for j in range(start, start + BATCH_SIZE):
            sample = train_data[j]
            input_enc.add(sample.sv_enc)

            input_dec.add(sample.sv_dec)

            label_enc.append(sample.value)

            label_dec.extend(sample.policy)

            for _ in range(len(sample.policy)):

                mask.append(1.0)

            for _ in range(64 - len(sample.policy)):

                mask.append(0.0)

                label_dec.append(0.0)

                input_dec.offset.append(len(input_dec.index))



        # Convert data to PyTorch tensors.

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

        mask_tensor = mask_tensor.view(BATCH_SIZE, -1)

        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device)

        label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)

        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device)

        label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)



        optimizer.zero_grad()



        # Get model predictions for the batch with AMP.
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out_enc, out_dec = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device))
            
            # Calculate loss.
            loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
            loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
            loss_dec = loss_dec * mask_tensor
            loss_dec = loss_dec.sum() / float(BATCH_SIZE)
            loss = loss_enc + loss_dec

        # Backpropagate the loss and update model parameters.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    scheduler.step()
    print(f"Training Finish. LR: {scheduler.get_last_lr()[0]:.6f}")


