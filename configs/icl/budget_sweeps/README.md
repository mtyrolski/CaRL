# Solver completion at action budgets

Run **relative** and **fixed-horizon AdaSubS-style** all-Luna conditions on the
same held-out boards and search seeds. Each displayed budget is a separate solver
run with that action cap; these are not reconstructed points from a single run.
The main pilot uses 8 full, trajectory-derived 12×12/4-box test boards, seeds
11/22/33, and budgets 16/32/64/128/256: **240 solver runs**, plus calibration.
The `transfer.json` preset uses 8 synthetic 32×32/10-box test boards. Its results
must be reported separately from the 12×12 data.

## Inspect the plan — no API

From the CaRL root, with `uv sync --frozen --extra icl --group lint`:

```bash
uv run --extra icl python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/pilot.json --plan
```

`--plan` prints the exact condition/board/seed/budget count; it does not read
`.env`, calibrate or make inference requests. Review the JSON configurations
and their inherited limits before choosing `--execute`.

For a cheaper live interface check on two **full** test boards, use
`quick.json`: 2 boards × 1 seed × 2 methods × 2 budgets = 8 runs. It uses the
same model, prompts, examples, scales and calibration contract as the pilot.
Its small sample is not adequate for a stable performance conclusion.

`quick_coordinates.json` is a separate exploratory quick sweep with a
coordinate-augmented policy for both methods. First calibrate using
`quick_coordinates_relative.json`: a calibration for the original policy cannot
be reused. Keep its results separate from `quick.json`, which uses the original
policy. The live debugging also inspected test trajectories 199, 434 and 732;
exclude these layouts or prepare another disjoint holdout before making final
performance claims.

## Calibrate, execute, evaluate

The trajectory-tuned profile has a separate `tuned_quick.json` sweep and matching
fixed/relative conditions. Prepare value examples first, estimate tokens using
`configs/icl/token_estimates/tuned_assumptions.json`, and calibrate relative with
`tuned_quick_relative.json`. Its model uses reasoning tokens and may retry invalid
generated boards; those are explicitly represented in the tuned cost scenarios.
Neither condition runs a separate simulator search to verify generated subgoals.

**These live commands incur API usage. They are for you to run explicitly.**

```bash
# One separate, reusable calibration for the same policy and provider.
uv run --extra icl python -m scripts.run_icl_search calibrate \
  --config configs/icl/budget_sweeps/pilot_relative.json --execute

# Five independent action-budget settings; no implicit live mode.
uv run --extra icl python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/pilot.json --execute

# Alternatively, start with the eight-run quick check (after calibration):
# uv run --extra icl python -m scripts.run_budget_sweep \
#   --config configs/icl/budget_sweeps/quick.json --execute

# Local, offline analysis of the saved outcomes:
uv run --extra icl python -m scripts.evaluate_icl \
  --runs analysis/luna_icl/budget_sweep_pilot \
  --output analysis/luna_icl/budget_sweep_pilot_report
```

Open `analysis/luna_icl/budget_sweep_pilot_report/index.html` for the
**independently executed budgets** plot. `budget_sweep.csv/json` hold the points;
`pairs.csv/json` contain paired method comparisons for each action budget.
The existing per-cohort *observed search effort* curves remain available, but
are different: they summarize runs made at one cap.

The evaluation notebook, `notebooks/luna_icl_evaluation.ipynb`, accepts the same
input directory via `RUN_DIRS` and shows the new figure and coverage table.

## Restart, failures and provenance

Every trial writes an atomic detail file before its result row. After an
interruption, rerun the same command with `--resume`; completed trials are reused,
and a detail written immediately before an interruption is recovered. Only one
process can use a sweep output directory at a time (the lock uses Unix `flock`,
including under WSL). The manifest fingerprints the
resolved condition configs, provider, selected input files, calibration, relevant
code and package versions. A mismatch blocks resume. Do not edit inputs or code
mid-sweep; start a new output directory if you need different settings.

An `api_error` is saved and stops the sweep; it is not silently retried or
excluded. Inspect the saved error and use a **new** output directory for a fresh
attempt. Completed request-cache entries can be reused in the new attempt, so
physical usage and timing will differ; preserve both directories for accounting.
`api_budget` and `expansion_budget` are recorded as failures and are counted in
the results. A whole budget with no saved runs cannot appear in the plot; compare
coverage against `sweep_plan.json`. An open marker indicates incomplete saved
trials, a missing condition or an entirely unstarted planned budget. Partial points use observed trials
only and should not be treated as complete comparisons.

The cache is shared across budgets, methods and search seeds for identical
requests. Logical request/token budgets count cache hits; physical request and
token usage reflect only provider calls made in that particular run. Thus
physical cost and wall time depend on cache warm-up and method execution order.
Search seeds repeat the controller's categorical sampling; they are **not**
independent model generations or independent trained components. A model alias,
provider-side changes, API failures, interruptions and cache deletion can affect
reproducibility. The manifest is a guard, not a guarantee of identical live runs.

The configured per-run limits apply in addition to the action budget: 512 logical
requests, 32,768 output tokens, 1,000,000 estimated input tokens, 256 expansions
and 1,800 seconds. A run may therefore stop before reaching its action cap. The
calibration has separate limits: up to 4,096 logical policy requests and 14,400
seconds. The largest pilot has at most 240 × 512 logical search requests, plus
calibration; actual provider calls/tokens depend on cache and model outputs.
No API prices are assumed. Begin with a smaller explicitly edited copy if needed.

## Pilot, validation selection and transfer

The supplied scales `u=[0.9,0.5]` and `k=[8,4]` are **pilot_unvalidated**. They
are not a final model-selection claim. The fixed method uses horizon cap `k`;
the relative method uses the separate shared cap `L=16`. Consequently this
comparison changes both scale semantics and cap, not only the label.

For a more defensible final test, first perform matched-budget scale selection
on the disjoint validation layouts (also costs API usage):

```bash
uv run --extra icl python -m scripts.run_icl_search tune \
  --config configs/icl/budget_sweeps/pilot_relative.json --execute \
  --output analysis/luna_icl/budget_sweep_tuning

# The tuner writes the selected_relative.json and selected_fixed.json used below.
uv run --extra icl python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/selected_12.json --plan
uv run --extra icl python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/selected_12.json --execute
```

The selected config freezes the chosen scales before the test sweep. Selection
itself is an experiment with extra costs; a small validation set is not proof of
optimality. `selected_transfer.json` uses the same selected scales for the
synthetic 32×32 transfer family after tuning. Pilot transfer is available through
`transfer.json`, with the same separate calibration if the policy contract is
unchanged. Select the appropriate result directory when evaluating.

## Offline plumbing check

```bash
uv run --extra icl python -m scripts.run_icl_search calibrate \
  --config configs/icl/budget_sweeps/mock_relative.json --mock
uv run --extra icl python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/mock.json --mock
uv run --extra icl python -m scripts.evaluate_icl \
  --runs analysis/luna_icl/budget_sweep_mock \
  --output analysis/luna_icl/budget_sweep_mock_report
```

This uses two real full 12×12 boards and budgets 4/8, but heuristic mock
components and one seed. **It is not a Luna performance result.** Mock and live
calibration/cache records are separate.

## Statistical interpretation

Each point averages success across available seeds within each board, then gives
each board equal weight. The 95% interval bootstraps boards, retaining their
seeds together; one board has no interval and small samples can yield degenerate
intervals. Relative vs fixed differences pair identical `(board, seed)` within
each budget. Report sample coverage and API/expansion stops with every plot.
Separate reruns can be non-monotone because truncated controller rollouts can
alter the frontier and model requests. Do not draw conclusions from partially
completed budgets, zero-width intervals on tiny samples, or mock runs.

## Token and cost planning

Estimates are **offline workload scenarios**, not measured Luna usage or a
statistical expected value. Regenerate them from the actual selected boards,
prompts and editable assumptions before launching:

```bash
uv run --extra icl python -m scripts.estimate_icl_tokens \
  --sweep configs/icl/budget_sweeps/quick.json \
  --output analysis/luna_icl/token_estimates/quick
uv run --extra icl python -m scripts.estimate_icl_tokens \
  --sweep configs/icl/budget_sweeps/pilot.json \
  --output analysis/luna_icl/token_estimates/pilot
uv run --extra icl python -m scripts.estimate_icl_tokens \
  --sweep configs/icl/budget_sweeps/transfer.json \
  --output analysis/luna_icl/token_estimates/transfer
```

The resulting `estimate.md`, `estimate.csv` and `estimate.json` distinguish
calibration, search, model input and model output. Edit
`configs/icl/token_estimates/assumptions.json` for scenario workload fractions,
output-size multipliers or an updated price snapshot. The middle scenario
assumes each run uses half its action cap, four sampled policy steps per generator
call, one value request per generator call plus the root value, and half of the
maximum calibration policy calls. These are assumptions, not observations. It
assumes no primitive-action fallback or failed requests and does not simulate the
search frontier; actual counts and costs can be much higher or lower. Any scenario
exceeding configured limits is flagged, not silently clipped.

Each report includes one calibration. If calibration has already been completed,
use the separate **search** columns for incremental planning; when comparing quick,
pilot and transfer, do not add the shared calibration multiple times. Optional
validation tuning, repeated attempts, embeddings, notebook calls, regional
surcharges and taxes are excluded. Local `o200k_base` counts serialize the whole
request payload, matching the project's logical guard; provider-side tokenization
and billing may differ. The generator output is sized as a **complete explicit
board**, not a latent representation. Cache savings are not assumed; cache writes
and failed/restarted calls are not priced. The configured local accounting caps
are not a guaranteed provider-spend limit.

The default price snapshot is standard `gpt-6-luna` short-context pricing of
$0.10 per million uncached input tokens and $0.50 per million output tokens,
checked on 2026-11-02 in [OpenAI's model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
Prices, service tiers and regional terms can change: verify them before use.
The assumptions file records this source and date. No API calls or `.env` reads
are required to estimate; `selection=embeddings` and `num_subgoals > 1` are
rejected rather than undercounting their costs.
