import json

with open(r'c:\Users\Rahul\Downloads\90178661.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

last_valid_step = None
for step in reversed(data['steps']):
    for player_step in step:
        obs = player_step.get('observation', {})
        if obs.get('current') is not None:
            last_valid_step = step
            break
    if last_valid_step:
        break

if last_valid_step:
    p0_obs = last_valid_step[0].get('observation', {})
    p1_obs = last_valid_step[1].get('observation', {})
    
    current = p0_obs.get('current') or p1_obs.get('current')
    if current:
        players = current.get('players', [])
        for i, p in enumerate(players):
            print(f'--- Player {i} ---')
            print(f'Prizes: {p.get("prizes", [])}')
            print(f'Deck: {p.get("deck", [])}')
            active = p.get('active', [])
            print(f'Active: {active}')
            bench = p.get('bench', [])
            print(f'Bench: {bench}')
