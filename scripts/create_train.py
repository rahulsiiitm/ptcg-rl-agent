import os

with open(r'd:\Projects\4th Year\ptcg-rl-agent\scripts\user_pasted_code.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

train_lines = []
capturing = False
for line in lines:
    if line.startswith('device = torch.device('):
        capturing = True
    if capturing:
        train_lines.append(line)

train_code = ''.join(train_lines)

# We need to prepend the imports and model definition, which we can just import from hybrid_lucario
prefix = """import os
import sys
import glob
import math
import random
import torch

sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0] if glob.glob('/kaggle/input/**/cg-lib', recursive=True) else os.path.dirname(os.path.dirname(__file__)))

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
        sys.stderr.write(f"\\r{text} {percent}%   ")
        sys.stderr.flush()
        if(current >= count):
            sys.stderr.write("\\n")
            sys.stderr.flush()
            break
        yield current
        current += 1

"""

# Let's replace the top few lines in train_code that re-define device, model, optimizer etc.
# to use our imports and save properly.
# The user's script has:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyModel(128, 2, 256, 1, 1)
# model = model.to(device)

train_code = train_code.replace('torch.save(model.state_dict(), "out/model" + str(counter) + ".pth")',
                                'torch.save(model.state_dict(), "model_best.pth")')
train_code = train_code.replace('sample_deck = [721,721', '# sample_deck = [')

with open(r'd:\Projects\4th Year\ptcg-rl-agent\src\train\train_hybrid_mcts.py', 'w', encoding='utf-8') as f:
    f.write(prefix + train_code)

print('Training script generated successfully.')
