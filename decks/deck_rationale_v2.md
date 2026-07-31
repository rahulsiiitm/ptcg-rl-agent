# Phase 3 Deck Rationale: Snorlax Control

## Why Snorlax over Mono-Water?

In Phase 1 and 2, we utilized a Mono-Water Basic deck (Squirtle, Staryu, Poliwag). While extremely simple, it lacked a coherent win condition or disruption ability, causing the agent to stall out against opponents with higher HP or scaling attacks. 

The public ladder evidence strongly suggests that **simple, consistent, single-prize decks** perform best when piloted by a heuristic or RL agent. Complex combo chains (like ex/VSTAR engines) often break because the RL agent mis-sequences cards or plays suboptimal bench layouts.

## The Strategy

We are switching to a **Snorlax Control** deck. 
- **Main Attacker / Wall:** Snorlax (ID 1072). It boasts an incredible 160 HP for a Basic single-prize Pokémon.
- **Consistency / Engine:** Snorlax's ability "Gormandize" allows drawing up to 7 cards (ending the turn), ensuring the agent almost never dead-draws.
- **Disruption:** Boss's Orders (ID 1182) pulls up heavy retreat-cost opponent Pokémon, stranding them while Snorlax safely walls.
- **Sustainability:** Cheren (ID 1224) and Surfer (ID 1203) provide consistent draw support. Switch (ID 1123) acts as a pivot, and Ultra Ball (ID 1121) guarantees we find Snorlax on Turn 1.
- **Energy:** Basic Water Energy (ID 3) fulfills Snorlax's colorless attack cost.

## Decklist (`deck.csv`)
- 4x Snorlax (ID 1072)
- 4x Boss's Orders (ID 1182)
- 4x Cheren (ID 1224)
- 4x Surfer (ID 1203)
- 4x Switch (ID 1123)
- 4x Ultra Ball (ID 1121)
- 36x Basic Water Energy (ID 3)

## Matchup Profile
By having only a single Basic Pokémon (Snorlax), our agent cannot make "bad" active Pokémon choices during setup. Its game plan is highly linear: play Snorlax, attach energy, play Boss's Orders to trap opponents, and swing with high HP. This linearity heavily caters to PPO RL training, removing combinatorial branching from the initial turns.
