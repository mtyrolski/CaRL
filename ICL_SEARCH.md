# In-context search

CaRL can use language-model components to generate explicit Sokoban subgoals,
select low-level actions, and estimate state values. The search implementation
accepts simulator-backed components and records both high-level search and
primitive-action costs.

Full-state search conditions the controller on a complete target board.
Predicate search conditions the controller on a checked intent, with effects
and conditions that must hold during execution.

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

## Offline quick start

From the repository root, install the `icl` extra with the pinned CPython
3.11.4 and the repository lockfile. Add `--offline` to `uv sync` if Python
and the locked distributions are already available locally.

```bash
mkdir -p analysis/icl
work="$(mktemp -d "$PWD/analysis/icl/quickstart.XXXXXX")"
export UV_PROJECT_ENVIRONMENT="$work/venv"
uv sync --locked --no-dev --python 3.11.4 --extra icl

uv run --locked --offline --no-sync python -m scripts.run_icl_smoke \
  --config configs/icl/quickstart/sokoban_sweep.json --output "$work/sokoban"
uv run --locked --offline --no-sync python -m scripts.evaluate_icl \
  --matched "$work/sokoban/sweep.json" --output "$work/report"
```

The smoke prepares checked demonstrations and a frozen prompt, runs three
planning modes with matched tasks and budgets, and saves individual trials
for offline evaluation. It uses a bundled Sokoban fixture and requires no
credentials or model weights. The command reference also covers zero-shot
and four-shot prompts, OpenAI Responses and Ministral Chat Completions
fixtures, native Craftax on CPU, a local LoRA train-and-load run, and the
full-state reachability pilot. [Experiment commands](scripts/README.md).

## Full-state reachability (Figure 1)

The protocol tests the exact future board from a held-out trajectory. At
offset `k`, count the first match to that board within at most `k` attempted
actions, then compute pointwise 95% Wilson intervals. The full manifest
specifies 12×12 four-box boards for support or training and a disjoint set of
12×12 ten-box test boards, with 100 valid cases per full point. Models may
declare different offset subsets.

```bash
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_smoke.json --output "$work/figure1"
```

The smoke uses a deterministic fixture. The full campaign requires external
trajectories and pinned model artifacts. [Campaign inputs and invocation](scripts/README.md#full-state-reachability-figure-1).

## Inputs and results

Keep datasets, model weights, credentials, response caches and generated results
outside source control. The default local result directory is `analysis/`.
Check split separation, input provenance and controller configuration before
comparing methods. Model, prompt, decoding and action budget are part of the
experiment contract. Live inference requires an explicit flag, provider
credentials and a campaign ledger. [Provider setup](scripts/README.md#provider-profiles).

Run the relevant checks with `make test`, `make ruff` and `make pyright`.
