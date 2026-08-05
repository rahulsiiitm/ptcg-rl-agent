import os
import sys
import glob
import math
import random
import torch

sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0] if glob.glob('/kaggle/input/**/cg-lib', recursive=True) else 'd:/Projects/4th Year/ptcg-rl-agent/cg')

from cg.game import battle_start, battle_finish, battle_select
from src.agent.hybrid_lucario import (
    MyModel, LearnSample, LearnInput, mcts_agent, random_agent, 
    get_encoder_input, get_decoder_input, SparseVector
)

# Load the actual Lucario deck
deck_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deck.csv")
with open(deck_path, "r", encoding="utf-8") as f:
    sample_deck = [int(line) for line in f.read().splitlines() if line.strip()]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel(128, 2, 256, 1, 1)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function

loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function

os.makedirs("out", exist_ok=True)



# The main training loop.

for counter in range(5):

    torch.save(model.state_dict(), "model_best.pth")  # Save the current model.

    sample_list:list[LearnSample] = []  # List of training data samples.

    

    model.eval()

    with torch.inference_mode():

        # Evaluation

        results = [0, 0, 0]



        for i in progress(50, "Evaluating... "):

            obs, start_data = battle_start(sample_deck, sample_deck)

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

                    selected, _ = mcts_agent(obs, sample_deck, model)

                else:

                    selected = random_agent(obs)

                obs = battle_select(selected)

            

            battle_finish()  # Finalize the game.



            if obs["current"]["result"] == 2:  # Draw

                results[2] += 1

            elif obs["current"]["result"] == your_index:  # Win

                results[0] += 1

            else: # Lose

                results[1] += 1

        print("Evaluation win rate " + str(100 * results[0] // (results[0] + results[1])) + "%", flush=True)



        # Self Play

        for _ in progress(100, "Training Data Collecting... "):

            obs, _ = battle_start(sample_deck, sample_deck)

            samples:list[list[LearnSample]] = [[], []]  # [Player0 samples, Player1 samples]

            while True:

                if obs["current"]["result"] >= 0:

                    break



                # The MCTS agent generates an action and a training sample.

                selected, sample = mcts_agent(obs, sample_deck, model)

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

    random.shuffle(sample_list)

    BATCH_SIZE = 128

    batch_count = len(sample_list) // BATCH_SIZE

    for i in range(batch_count):

        # Prepare a batch of data.

        input_enc = LearnInput()

        input_dec = LearnInput()

        mask = []

        label_enc = []

        label_dec = []

        start = BATCH_SIZE * i

        for j in range(start, start + BATCH_SIZE):

            sample = sample_list[j]

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



        # Get model predictions for the batch.

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

        loss.backward()

        optimizer.step()

    print("Training Finish.")


