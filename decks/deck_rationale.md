# Deck Selection Rationale

Based on Phase 0 guidelines, we need simple, consistent, single-prize decks that an AI can pilot effectively without complex combo requirements.

## Phase 1 & 2: Mono-Water Aggro
Originally, we used a simple Mono-Water basic deck to ensure the rule-based agent and early RL iterations had an easy time piloting it (just attach {W} and attack). This provided our baseline score of 170.1 (Phase 1) and 303.6 (Phase 2).

## Phase 3: Snorlax Control
For Phase 3, we are upgrading to a **Snorlax Control** deck designed to exploit the RL agent's capability to learn positional and resource-advantage strategies.

### Deck List (snorlax_deck.csv):
- **4x Snorlax (1072)**: 160 HP single-prize wall. `Gormandize` draws cards, `Collapse` deals 160 damage (and puts Snorlax to sleep).
- **4x Boss's Orders (1182)**: Drags vulnerable opponent Pokémon to the active spot.
- **4x Cheren (1224)**: Simple +3 card draw.
- **4x Surfer (1203)**: Switch active and draw to 5 (great for a Sleeping Snorlax).
- **4x Switch (1123)**: Item-based switch to escape Sleep.
- **4x Ultra Ball (1121)**: Consistency to find Snorlax.
- **36x Basic Water Energy (3)**: Pays for Snorlax's `Collapse` ({C}{C}{C}) and guarantees attachments.

### Why Snorlax?
1. **High HP / Single Prize**: 160 HP is incredibly bulky for a single-prize Pokémon. It easily survives hits from many meta decks.
2. **Simple Gameplan**: The RL agent only has to learn: Attach -> Draw (Gormandize/Cheren/Surfer) -> Switch (if asleep) -> Attack (Collapse).
3. **Control Elements**: Boss's Orders allows the RL agent to learn target prioritization, trapping weak utility Pokémon while Snorlax heals or sets up.
4. **Engine Compatibility**: Validated against the `cabt` engine (errorType: 0). We use Basic Water Energy instead of Special Energy to avoid Turn 0 validation errors with the Kaggle environment.
