#!/bin/bash
# Validates the agent by running a match against itself locally.

echo "Running local validation match..."
python -c "
from kaggle_environments import make
from main import agent

with open('deck.csv') as f:
    deck = [int(line.strip()) for line in f.readlines() if line.strip()]

env = make('cabt', configuration={'decks': [deck, deck]})
env.run([agent, agent])

with open('eval/result.html', 'w') as f:
    f.write(env.render(mode='html'))
print('Simulation finished. Wrote eval/result.html')
"
