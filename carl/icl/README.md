# In-context Sokoban search

The public modules keep existing imports stable. Implementation details live in
packages where the components have separate responsibilities:

| Package or module | Responsibility |
| --- | --- |
| `llm_components/` | Model-neutral generator, goal-conditioned policy and value |
| `luna_components/` | Legacy imports and frozen formatting used by existing datasets |
| `checkpoints/` | Inspectable, fingerprinted generator demonstrations |
| `data_sources/` | Sokoban trajectory loading and replay validation |
| `planning/` | Search protocols, result records and queue entries |
| `evaluation/` | Offline tables, statistics, board graphics and Plotly figures |
| `experiments/` | Validation-only tuning |
| `api.py`, `responses.py` | Cached Responses API transport and streaming decoding |
| `campaign_budget.py` | Cross-process reservations and usage settlement |
| `search.py` | Simulator execution, scale/value priority and work limits |
| `luna.py`, `datasets.py` | Public facades for components and data preparation |

`action_budget` counts every executed low-level simulator action, including
failed rollouts, no-ops and fallback. `max_expansions` optionally limits queue
entries separately; set it to `None` for the CaRL-style single-budget comparison.
High-level states reached by a rollout have already consumed low-level steps.

The generator returns full explicit boards. Static rejection never asks the
simulator for a move. CLLP executes actions; a subgoal hit requires exact state
equality. The simulator determines whether the full Sokoban task is solved.
The experimental verbal controller uses its own model judge for subgoal stops,
while retaining the simulator's full-task success condition.

Run `make lint`, `make pylint-icl` and `make test` from the repository root.
The campaign launch guide describes execution, cost limits and resume behavior.
