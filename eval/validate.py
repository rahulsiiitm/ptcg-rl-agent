import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make
from main import agent

with open('deck.csv') as f:
    deck = [int(line.strip()) for line in f.readlines() if line.strip() and not line.startswith('#')]

env = make('cabt', configuration={'decks': [deck, deck]})
env.run([agent, agent])

with open('eval/result.html', 'w', encoding='utf-8') as f:
    f.write(env.render(mode='html'))
print('Simulation finished. Wrote eval/result.html')
