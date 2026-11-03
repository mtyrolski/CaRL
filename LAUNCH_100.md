# Launch 100 boards with a 500-node budget

Run commands from the repository root. This is a new experiment with the measured
goal-placement prompt; it does not resume the historical pilot on changed code.

| Preset | Conditions | Planned runs |
| --- | --- | ---: |
| `core` | `relative_u`, `absolute_d` | 200 |
| `extended` | Core plus `fixed_k`, `verbal_relative` | 400 |
| Trained reference | Local Torch components in the shared search loop | 100 |

All conditions use the same 100 full, distinct **12×12×4** initial boards.
No 32×32 full solves or holdout1000 runs are included.

The node limit means **500 expanded queue entries**, including entries for
different scales and primitive fallback. Discovered/proposed nodes are a separate
quantity. Every simulator action counts against an additional **4,000-action**
ceiling, including no-ops and unsuccessful rollouts. Seed: 11. CLLP cap: 8;
the fixed baseline uses cap=k. Explicit scales are u=(0.85, 0.5) and d=(8, 4),
carried forward from validation, not retuned for the new prompt.

## Inspect and freeze without API calls

```bash
uv sync --frozen --extra icl
uv run --extra icl python -m scripts.launch100 --plan
uv run --extra icl python -m scripts.launch100 --prepare
uv run --extra icl python -m scripts.launch100 --status
```

Default invocation is also a plan. The plan lists every board ID, resolved
methods and limits, remaining ledger balance, and explicit cost assumptions.
Preparation writes `analysis/luna_icl/launch100_nodes500/manifest.json` and an
all-planned-cells `status.csv`. It does not infer. The manifest binds source
hashes, model, inputs, exact prompts and examples. Changed source or configuration
requires a new output directory; a different freeze is never silently resumed.

Inspect versioned inputs under
[`configs/icl/checkpoints/launch100/`](configs/icl/checkpoints/launch100/README.md).
`assets/config.json` holds the full explicit prompts; `assets/demonstrations.json`
contains six goal-placement demonstrations; `assets/checkpoint/generator.json`
contains 26 measured scale examples. `assets/verbal.json` contains the verbal
generator, controller, judge and six measured semantic calibration examples.
The explicit launcher inserts both demonstration sets. Test manifests contain
initial states and provenance, without witness actions or future test states.

## Cost and execution

**The existing $40 campaign cap is not enough to promise a complete matrix.**
Extrapolating the one measured goal-progress full-board probe gives approximately
$249 for 200 runs at the same cost per expansion, or $561 at the same cost per
action. These are scenarios from one partial, mixed-cache validation run, not
quotes or measured costs of this experiment. They assume the same cost for both
explicit methods. The plan includes token/request/time scenarios. Verbal full-run
cost is not established. No new experiment charges were incurred during setup.

The shared ledger retains all previous charges and unknown reservations.
Relative and absolute stages each have a $12 allocation; fixed and verbal share
the existing $4 baseline allocation. Lower account/repository limits still apply.
`--usd-cap 40` means the **absolute ceiling for the existing campaign**, not $40
of fresh funding. A lower cap is persistent. The launcher refuses execution when
the existing ledger is missing; do not start a fresh ledger to bypass old usage.

Place the key in the repository `.env`. This command deliberately clears an
inherited key so the file is used:

```bash
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.launch100 --execute --usd-cap 40
```

Resume the same frozen experiment:

```bash
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.launch100 --execute --usd-cap 40 --resume
```

Execution is serial, alternating methods within each board. A second launcher on
the same output is blocked by its lock. Each call has a 90-second timeout; each
process invocation has a six-hour per-run wall limit. Cumulative logical request
and token counters survive resume. Transport gets at most one additional attempt
for an explicit 429/5xx response, with bounded backoff and a separately reserved
cost. Authentication errors and uncertain timeouts stop execution. Responses and
actual action prefixes are journaled; a pending request can resume only when its
completed cache entry exists. Completed solutions and action traces are replayed
before acceptance. API failures remain visible; missing cells remain missing.

## Optional conditions and trained checkpoints

The extended preset needs its own output directory and freeze:

Choose it instead of `core` when starting all four Luna conditions. It includes
the two core methods again; it reuses matching request-cache entries, not completed
run records from a different freeze.

```bash
uv run --extra icl python -m scripts.launch100 --preset extended --output analysis/luna_icl/launch100_extended --prepare
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.launch100 --preset extended --output analysis/luna_icl/launch100_extended --execute --usd-cap 40
```

The explicit fixed baseline uses generators at k=8 and 4 plus enumerated primitive
fallback. It is not the original native k=8,4,1 solver. The verbal condition uses
a separate model judge for subgoal completion; full-task success still requires
the simulator's solved predicate. Its tiny calibration and post-hoc prompt
selection limit interpretation.

For local trained models, inspect
[`configs/trained_reference/launch100.json`](configs/trained_reference/launch100.json).
The preset points to existing local checkpoints; it never downloads weights.

```bash
uv run --extra icl python -m scripts.trained_reference.run plan
uv run --extra icl python -m scripts.trained_reference.run run --execute
uv run --extra icl python -m scripts.trained_reference.run run --execute --resume
uv run --extra icl python -m scripts.trained_reference.run status
```

This trained comparison uses the shared search loop with the same expansion and
action limits, scales 8/4, one proposal and four beams. It is separate from the
historical native AdaSubS result of 91/100, which used a different cohort and
native work counter. Trained resume verifies completed cases; an interrupted
case is retained for inspection rather than silently rerun.

## Offline tables and figures

```bash
uv run --extra icl python -m scripts.launch100.report --static
```

`analysis/luna_icl/launch100_nodes500/plot_data/` contains `runs.csv`,
`solve_rates.csv`, `paired_delta.csv`, `COLUMNS.md`, and HTML/SVG/PDF figures.
Pass `--output` to report another prepared preset. Export works before any run:
missing `solved` values stay NULL, and figures state that no results exist.
Solve-rate denominators use observed boards; coverage and possible full-cohort
bounds remain separate. Zero observed successes retain a nonzero Wilson upper
bound. Ledger charges and unresolved reservations are separate columns.

```python
import pandas as pd

runs = pd.read_csv("analysis/luna_icl/launch100_nodes500/plot_data/runs.csv")
observed = runs.loc[runs["status"].eq("complete")]
paired = observed.pivot(index="board_id", columns="method", values="solved")
paired = paired.dropna(subset=["relative_u", "absolute_d"])
delta_pp = 100 * (paired["relative_u"].astype(float) - paired["absolute_d"].astype(float)).mean()
```

Quality checks: `make lint`, `make pylint-icl`, `make test`. Historical one-off
tools are documented in `scripts/archive/campaign_20261102/README.md`; they are
not additional prerequisites for this launch.
