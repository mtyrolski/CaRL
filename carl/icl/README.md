# In-context Sokoban search

The public modules keep existing imports stable. Implementation details live in
packages where the components have separate responsibilities:

| Package or module | Responsibility |
| --- | --- |
| `luna_components/` | Generator, goal-conditioned policy, value, message formatting |
| `checkpoints/` | Inspectable, fingerprinted generator demonstrations |
| `data_sources/` | Sokoban trajectory loading and replay validation |
| `planning/` | Search protocols, result records and queue entries |
| `evaluation/` | Offline tables, statistics, board graphics and Plotly figures |
| `experiments/` | Validation-only tuning |
| `api.py`, `responses.py` | Cached Responses API transport and streaming decoding |
| `campaign_budget.py` | Cross-process reservations and usage settlement |
| `search.py` | Simulator execution, scale/value priority and work limits |
| `luna.py`, `datasets.py` | Public facades for components and data preparation |

`max_expansions` counts queue entries selected for expansion. It does not count
all proposed or discovered states. `action_budget` separately counts every
executed simulator action, including failed rollouts, no-ops and fallback.

The generator returns full explicit boards. Static rejection never asks the
simulator for a move. CLLP executes actions; a subgoal hit requires exact state
equality. The simulator determines whether the full Sokoban task is solved.
The experimental verbal controller uses its own model judge for subgoal stops,
while retaining the simulator's full-task success condition.

Run `make lint`, `make pylint-icl` and `make test` from the repository root.
The campaign launch guide describes execution, cost limits and resume behavior.
