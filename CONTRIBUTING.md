# Contributing to PTCG AI Battle Agent

Thanks for taking an interest in the project. Contributions are welcome across game AI, reinforcement learning, search, evaluation, replay analysis, testing, and documentation.

## Before you start

For substantial changes, open an issue or join a Discussion first. This is especially useful for changes to agent behaviour because a locally sensible rule can create regressions in other matchups.

Good contributions are evidence-driven. If you are changing how the agent plays, include the replay, scenario, benchmark, or experiment that motivated the change.

## Development setup

```bash
git clone https://github.com/rahulsiiitm/ptcg-rl-agent.git
cd ptcg-rl-agent
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

See the README and `docs/` for project-specific workflows and evaluation notes.

## Ways to contribute

- Fix a reproducible gameplay or engine-safety bug
- Add tactical regression tests from real replay failures
- Improve replay/evaluation tooling
- Explore RL, MCTS, imitation learning, or opponent modelling
- Improve deck/general-policy abstractions
- Add documentation for experiments and failure modes

## Behaviour changes

When changing agent behaviour, please include:

1. **Problem**: what decision or matchup is failing?
2. **Evidence**: replay, scenario, or reproducible state.
3. **Change**: what logic or model behaviour changed?
4. **Evaluation**: what tests or games were run?
5. **Trade-offs**: what could regress because of the change?

Do not report simulated improvements as real ladder improvements. Keep tactical tests, simulated benchmarks, and Kaggle ladder results clearly separated.

## Pull requests

Keep PRs focused. Avoid mixing unrelated refactors with policy changes.

Before submitting:

- Run the relevant tests/evaluation.
- Confirm the agent still produces legal actions.
- Check for crashes/timeouts if the execution path changed.
- Add or update tests for fixed regressions when practical.
- Update documentation if results or public behaviour changed.

A failed experiment can still be valuable. If the result teaches us something, document it rather than hiding it.

## Research contributions

Research issues are intentionally open-ended. You do not need to arrive with a finished architecture. Small reproducible experiments with clear hypotheses and results are useful.

When sharing experimental numbers, include enough information to reproduce or interpret them: agent/deck version, opponent policy, number of games, seeds/settings when relevant, and whether the result came from simulation or the real ladder.

## Community

Be constructive when reviewing ideas and results. Challenge assumptions, not people. Pokémon battles can be ruthless; the Discussions tab does not need to be.
