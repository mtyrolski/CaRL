# In-context search

CaRL can use language-model components to generate explicit Sokoban subgoals,
select low-level actions, and estimate state values. The search implementation
accepts simulator-backed components and records both high-level search and
primitive-action costs.

The explicit-state generator proposes a complete target board. A controller
conditioned on that board is not automatically conditioned on a structured
intent. Structured intent utilities are provided separately and support
predicate validation and checked trajectory annotations.

## Components

| Module | Responsibility |
| --- | --- |
| `carl.icl.llm_components` | Generator, controller and value interfaces |
| `carl.icl.planning` and `carl.icl.search` | Search state, queues and simulator execution |
| `carl.icl.calibration` | Controller duration and scale calibration |
| `carl.icl.data_sources` and `carl.icl.datasets` | Board and trajectory preparation |
| `carl.icl.intent_data` and `carl.icl.intent_augmentation` | Structured predicates and coordinate transformations |
| `carl.icl.evaluation` | Offline summaries, statistics and plots |

Search counts all attempted primitive actions, including unsuccessful actions
and no-ops. Expansion limits, model calls and wall time are recorded separately.
A target-board hit is distinct from task completion.

## Running experiments

Install the optional ICL dependencies using the project's package manager, and
run commands from the repository root. Start with an offline preview:

```bash
uv run --extra icl python -m scripts.run_icl_search preview
uv run --extra icl python -m scripts.run_budget_sweep --config configs/icl/budget_sweeps/mock.json --plan
```

`run_icl_search` provides `prepare`, `prepare-value`, `preview`, `calibrate`,
`run`, `matrix` and `tune` commands. `--mock` uses local test components;
`--replay` reads cached responses; `--execute` enables live API requests.
Each experiment requires its own input data and configuration.

The OpenAI and replay providers also require a local
`configs/icl/campaign_budget.json` and a shared ledger. The example budget
configuration has a zero limit and no model prices, so it cannot authorise paid
requests. Configure an appropriate ceiling, stage allocations and current
pricing before use. Reuse the same ledger across a campaign and reconcile
unknown attempts before continuing. Once a ledger exists, its configuration
must remain byte-identical; changing it is rejected.

For saved runs, `scripts.evaluate_icl` produces offline summaries. See
[`scripts/README.md`](scripts/README.md) and the configuration directories for
command options and examples.

The hybrid full-state runner accepts `--prepare-data` to build its local
demonstrations and evaluation inputs from configured trajectory data. The
structured-intent generator exposes training and evaluation commands via
`python -m carl.icl.intent_generator_experiment --help`.

## Inputs and results

Keep datasets, model weights, credentials, response caches and generated results
outside source control. The default local result directory is `analysis/`.
Check split separation, input provenance and controller configuration before
comparing methods. Model, prompt, decoding and action budget are part of the
experiment contract.

Run the relevant checks with `make test`, `make ruff` and `make pyright`.
