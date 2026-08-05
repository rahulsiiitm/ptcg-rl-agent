import ast, os

# 1. Syntax check
with open('src/agent/rule_based_lucario.py', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print('[1] SYNTAX OK')
except SyntaxError as e:
    print('[1] SYNTAX ERROR:', e)

# 2. Check N_DET, K_OPP values
n_det = [l for l in src.split('\n') if 'N_DET' in l and '=' in l and l.strip().startswith('N_DET')]
k_opp = [l for l in src.split('\n') if 'K_OPP' in l and '=' in l and l.strip().startswith('K_OPP')]
print('[2] N_DET:', n_det[0].strip() if n_det else 'NOT FOUND')
print('    K_OPP:', k_opp[0].strip() if k_opp else 'NOT FOUND')

# 3. Check candidate pool size
cand_line = [l for l in src.split('\n') if 'len(cand) >= 8' in l]
print('[3] Candidate pool widened to 8:', 'OK' if cand_line else 'NOT FOUND')

# 4. Check archetype matching
print('[4] _match_archetype:', 'OK' if '_match_archetype' in src else 'NOT FOUND')

# 5. Check Team Rocket Energy override
print('[5] Team Rocket override:', 'OK' if '_TEAM_ROCKET_ENERGY_ID' in src else 'NOT FOUND')

# 6. Check static templates
print('[6] Grimmsnarl static:', 'OK' if '_GRIMMSNARL_LINE' in src else 'NOT FOUND')
print('    Dragapult static:', 'OK' if '_DRAGAPULT_LINE' in src else 'NOT FOUND')

# 7. Check top20_decks directory
templates = [f for f in os.listdir('top20_decks') if f.endswith('.csv')]
print('[7] top20_decks/ templates:', len(templates), 'files')
for t in sorted(templates):
    n = len(open('top20_decks/' + t).read().strip().split('\n'))
    print('   ', t + ':', n, 'cards')

print('\nAll checks done.')
