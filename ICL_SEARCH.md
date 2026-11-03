# All-Luna ICL search

The current supported launch is [LAUNCH_100.md](LAUNCH_100.md): 100 distinct
12×12×4 boards, 500 expansions, matched relative/absolute conditions, and
optional fixed, verbal and trained-model comparisons. Its portable inputs and
explicit cost guard replace the earlier default pilot instructions below.
The remainder documents generic interfaces and historical experiment profiles.

Generator, conditional policy and value all use `gpt-6-luna` through Responses
with JSON answers and few-shot examples. No local model weights are loaded,
and no component is trained. The policy returns four action probabilities;
the executor samples one action and checks it in the exact CaRL simulator.

This implements the proposal's **LLM / explicit-state experiment**. It compares
policy-relative percentile conditioning with fixed-horizon AdaSubS-style search.
It does not implement the learned latent encoder or latent MSE training from
the main method. See [the proposal audit](analysis/luna_icl/PROPOSAL_ALIGNMENT.md).

## Launch

For independent solver runs at multiple action budgets, see the ready-to-run
[budget sweep configurations and instructions](configs/icl/budget_sweeps/README.md).
The default 12×12 pilot pairs fixed and relative methods across eight held-out
boards and three search seeds; it does not start until explicitly invoked.

For a tiny interactive walkthrough, open
[`notebooks/luna_components_demo.ipynb`](notebooks/luna_components_demo.ipynb)
with the project `.venv/bin/python` kernel. It shows the generator's subgoal,
policy probabilities, sampled moves, board transitions and value estimates.
It loads a full held-out `real_12x12x4` board and examples from four training
layouts in the prepared `data/icl` dataset. The tuned value uses four states
per layout. Set `BOARD_INDEX` to choose a board, or `FAMILY`
to `generated_32x32x10` for synthetic transfer boards. `PROVIDER = 'mock'` keeps
inference offline on those same boards; set it to `'openai'` for Luna. The client
has a 16-request logical budget; the default four-step walkthrough uses at most
nine requests including up to three generator attempts. The full-search section uses a separate bounded client. It runs
automatically with mock inference; set `RUN_SEARCH = True` to opt in with Luna.
Its action, expansion and logical-request budgets are editable in that cell.
The Plotly tree shows actual attempt endpoints, including rejected and duplicate
leaves, rather than only frontier states; repeated scales can expand the same
path more than once. Select `NODE_ID` to inspect a target and actual endpoint,
and `REPLAY_STEP` to examine a simulator-verified solution. The separately
labelled mock-only one-move fixture checks a successful replay and is not a
full-board benchmark.
The fixed-horizon walkthrough needs no calibration; the relative option requires
matching completed calibration. Run `make icl-prepare` if data is absent.
The notebooks have been executed offline with mock inference
from the `2026/` workspace; API access and `.env` loading were blocked during
verification. Startup also has regression coverage from the repo and notebook
directories. If automatic discovery fails, set `REPO_PATH` in its first code cell.

## Trajectory-tuned components

`configs/icl/experiments/tuned_fixed.json` and `tuned_relative.json` are
exploratory profiles from live trajectory tests. The policy receives explicit
board coordinates and uses `low` reasoning. The generator returns complete
boards using a layout-specific JSON schema and can retry malformed proposals up
to three times. These checks inspect the returned board's structure; they do
**not** run a separate simulator search or replace the model's board. A valid
board may still be unreachable by the policy.

Value sees training states sampled from verified trajectories and predicts a
remaining move count; code converts it to `discount ** count`. Board-derived
features, including coordinates and Manhattan box–goal distance, are included
in the prompt. The model also supplies the terminal value: there is no terminal
override. Reference trajectories are not necessarily optimal, so their remaining
length is a useful comparison target, not ground-truth optimal distance.

Prepare the value demonstrations once using the existing shard, without model
calls or old generators:

```bash
uv run python -m scripts.run_icl_search prepare-value --config configs/icl/experiments/tuned_fixed.json
uv run python -m scripts.estimate_icl_tokens \
  --sweep configs/icl/budget_sweeps/tuned_quick.json \
  --assumptions configs/icl/token_estimates/tuned_assumptions.json \
  --output analysis/luna_icl/token_estimates/tuned_quick
```

For diagnostics, the commands below evaluate on two **validation** layouts.
`--execute` makes paid calls; use `--mock` for an offline wiring check. The
output directory is new and each component has its own bounded client. The
reference actions construct evaluation targets and labels only; they are never
given as an answer to a query.

```bash
uv run python -m scripts.evaluate_trajectory_components --component policy --split validation --indices 0 1 --horizons 4 8 --positions start --max-requests 30 --output analysis/luna_icl/trajectory_demo --execute
uv run python -m scripts.evaluate_trajectory_components --component generator --split validation --indices 0 1 --horizons 4 --positions start --max-requests 6 --output analysis/luna_icl/trajectory_demo --execute
uv run python -m scripts.evaluate_trajectory_components --component generated_policy --generator-report analysis/luna_icl/trajectory_demo/generator.json --max-requests 8 --output analysis/luna_icl/trajectory_demo --execute
uv run python -m scripts.evaluate_trajectory_components --component value --split validation --indices 0 1 --max-requests 10 --output analysis/luna_icl/trajectory_demo --execute
```

Open [`notebooks/luna_trajectory_diagnostics.ipynb`](notebooks/luna_trajectory_diagnostics.ipynb)
and set the report paths to inspect exact target hits, actual generated-target
rollouts, boards and value curves. The notebook makes no API calls. A single
trajectory or search seed is not a reliable estimate of general success; use
fresh disjoint layouts and enough trials for a final claim.

From this repository:

```bash
uv sync --frozen --extra icl --group lint
cp .env.example .env
# Put your key in .env: OPENAI_API_KEY=...
make icl-prepare
make icl-preview
make icl-smoke
```

`prepare` is local and deterministic. The data is already prepared in this
checkout; regenerate it after changing the data configuration. `preview` writes
prompts without network inference. Relative generator prompts require calibrated
labels, so a preview before calibration explicitly reports that part unavailable.
`icl-smoke` calibrates and runs **heuristic mocks**, never OpenAI.

For real inference, these commands explicitly opt into API charges:

```bash
make icl-calibrate
# Two easy held-out boards per method, one seed; interface check only.
uv run --extra icl python -m scripts.run_icl_search matrix --execute \
  --matrix configs/icl/experiments/matrix_smoke.json
# Full initial states: relative vs fixed, 12x12x4 and 32x32x10.
make icl-pilot
```

The pilot has 4 instances × 3 seeds × 2 methods × 2 families = 48 searches.
It uses pilot scales, not validated optimal scales. Start with the smoke run.
Calibration is separate: at most 8 contexts × 8 targets × 4 trials × 16 actions
= 4,096 policy interactions. Exact-hit and terminal stopping can reduce this.
No live OpenAI calls were made during implementation; account access, model
availability and actual model success still need the real run.

`.env` is ignored by Git. Exported environment variables take precedence.
Placeholder/missing keys fail before inference. `--model` overrides **all three**
roles and invalidates previous calibration. `--replay` uses only completed live
response cache entries; it neither sends requests nor falls back to mock data.
Run `uv run --extra icl python -m scripts.run_icl_search --help` for commands.

## Configurations and comparison

The editable generator, policy, value and scale prompts are in
`configs/icl/experiments/base.json` under `prompts`. See
[`configs/icl/prompts/README.md`](configs/icl/prompts/README.md) for overrides,
few-shot data, preview commands and calibration provenance. The demo notebook
loads the same configuration rather than a separate set of hardcoded prompts.

### Responses request contract

`carl/icl/responses.py` centralizes explicit message blocks and streaming options.
Developer/user content uses `input_text`; assistant few-shot answers use
`output_text`. The final message is the current user query. Requests use
`service_tier="default"`, low verbosity, reasoning effort `none` by default,
reasoning mode `standard`, summary `auto`, no tools, streaming and `store=False`.
The requested includes are `reasoning.encrypted_content` and
`web_search_call.action.sources`; these do not enable web search.

The default `response_format="text"` matches the supplied API examples. The
developer prompt includes the expected JSON schema, and component outputs are
parsed and checked locally. Set `response_format="json_schema"` to opt into strict
structured output. Both modes retain the component-specific output token limits.
The stream collector uses the completed response's full text and usage, closes
the stream on every exit, and rejects errors, refusals, incomplete responses and
streams that end without completion. Partial text is never passed to the solver.
Provider error messages are not copied into logs. Explicit prompt-cache extension
fields are no longer sent. The new request contract changes cache keys and policy
fingerprints, so previous calibration must be regenerated before new inference.

The streaming path is checked with a local HTTP mock fixture, and the component
notebook is checked end to end with mock inference. No real API calls were made
and `.env` was not read during these checks.

Configs are validated JSON with relative `extends`, recursive dictionary merging,
unknown-key rejection and value checks. File paths inside a config resolve from
the repository root. All prepared matrices enforce matched budgets, model,
few-shot count, seeds and evaluation instances within each board family.

| Config | Generator scale | Controller cap |
|---|---|---|
| `relative.json` | Success-weighted local percentile `u`: 0.9, 0.5 | 16 actions |
| `fixed.json` | Constant trajectory offsets `k`: 8, 4 | `k` actions |
| `absolute.json` | Measured mean successful first-hit time `d`: 8, 4 | 16 actions |
| `fixed_shared_cap.json` | Constant `k`: 8, 4 | 16 actions |
| `relative_32.json`, `fixed_32.json` | Same frozen scales and examples | As above |

All are in `configs/icl/experiments/`. The fixed baseline is an **ICL
reimplementation of the AdaSubS scheduling rule**, not a reproduction of the
original trained model's results. Its instructions and every generator example
use the requested constant `k`, with a simulator-verified `k`-action transition.
Each state enters the common priority queue at both scales and at primitive
fallback zero. Priority is lexicographic `(scale, value)`, largest first.
Zero is never sent to the generator. Shorter paths reopen states.

The generator uses four distinct training contexts. For calibrated methods,
one target per context is sampled with its success weight, deterministically;
relative and absolute-time conditions select the same examples and change only
their labels. Policy examples cover varied future horizons and supervise the
next action. Value examples use `0.99 ** remaining_solution_actions`.
Policy and value prompts/examples are identical across methods and transfer sizes.

Calibration uses distinct future states sampled uniformly within `D_ref=16`.
Four independent categorical action rollouts estimate exact-hit probability and
mean successful first-hit time. Targets with zero hits are discarded. Tied times
share the midpoint of their combined weighted CDF interval. Calibration is done
only on training layouts, never reconstructed for the test query.

`selection: "direct"` uses deterministic context order. `"embeddings"` retrieves
training contexts with `text-embedding-3-small`; it does not create a latent
search space. To compare this mode, run:

```bash
uv run --extra icl python -m scripts.run_icl_search matrix --execute \
  --matrix configs/icl/experiments/matrix_embeddings.json
```

The policy stays the same in direct/embedding experiments, so calibration can be
reused. Embedding requests count against inference request and input budgets.
Mock embedding retrieval is deliberately unavailable; preview shows unresolved
direct ordering and labels it as such.

## Validation selection and transfer

Use equal three-point grids on the four **validation** boards before reporting
transfer performance:

```bash
uv run --extra icl python -m scripts.run_icl_search tune --execute \
  --output analysis/luna_icl/selected
uv run --extra icl python -m scripts.run_icl_search matrix --execute \
  --matrix analysis/luna_icl/selected/matrix_selected.json
```

The configurable `tuning` grids in `base.json` default to relative `(0.9,0.5), (0.75,0.25), (0.95,0.65)` and fixed
`(8,4), (12,4), (16,8)`. Selection maximizes validation completion, breaks ties
with fewer attempted actions, then uses grid order. The command writes selected
12×12 and 32×32 configs, unchanged components/scales, raw outcomes and a manifest.
Mock tuning keeps the `pilot_unvalidated` label. API errors prevent selection.
Window, execution cap, prompts and few-shot count are fixed pilot choices here;
the full proposal also calls for their validation study.

To test intermediate generated sizes, inherit either method config and set
`family` to `generated_12x12x4`, `generated_12x12x10`, `generated_16x16x4`,
`generated_24x24x7`, or `generated_32x32x10`. Use a distinct `name`.
`method: "primitive"` and `"direct"` provide diagnostic nonhierarchical controls.
Ready configs for every generated family are also supplied in
`matrix_transfer.json` (144 searches with the default instance/seed counts).
`matrix_ablation.json` adds measured absolute time and the shared-cap fixed control.

## Data and reproducibility

The input shard is
`rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl`.
It contains 1,000 trajectories. Preparation selects and replays 20 complete
successful trajectories: 8 training contexts, 4 validation and 8 test boards.
Splits share no wall/goal layout, including the player/box-independent layout hash.

Preparation also makes 4 validation + 8 test boards for each of five synthetic
families listed above, for **80 unique layouts total**. The independent Python
generator scrambles solved boards through reverse pulls on connected random-walk
floors. Every board has a forward solution witness replayed in CaRL. It supports
32×32×10 without invoking or rebuilding the old C++ generator. Witness length
is not optimal difficulty; these layouts are a separate synthetic distribution.

Only the training file enters few-shot prompts. Test/validation public files
contain initial boards, identifiers and layout hashes. Witness actions live in
`data/icl/private_witnesses/` and are never loaded by inference. `_easy` files
contain one-move-to-solved fixtures and must not be treated as benchmarks.
Generated data, calibration and request caches are local, ignored artifacts;
the command recreates them from the input shard and recorded seed.

[Source inventory](analysis/luna_icl/source_inventory.csv) inspects the first
board in all 56 progress shards: these are 12×12 with 4–10 boxes, not 32×32.
[Dataset manifest](analysis/luna_icl/dataset_manifest.json) records source hash,
replayed transitions, split sizes, generation seeds and witness lengths.
The original data/checkpoint symlinks remain read-only inputs to this workflow.

## Budgets, caches and artifacts

Every attempted simulator action counts, including no-ops and failed controller
attempts. Task completion is checked after each action, even before a target
match. Only exact matched endpoints enter the frontier. The remaining budget
caps each rollout; no expansion can overshoot it.

Per-search defaults: 256 actions, 128 expansions, 512 logical API requests,
1,000,000 estimated input tokens total, 16,000 per request, 32,768 output tokens,
1,800 seconds. `max_output_tokens` in `limits` is the **total** response budget;
`generator_tokens`, `policy_tokens`, and `value_tokens` are per-response caps.
Calibration has its own limits in the same config. A full response allowance is
reserved before a request; an uncertain API failure conservatively consumes that
allowance. SDK retries are disabled. Timeouts and all available actual usage are
recorded, including incomplete Responses usage.
Wall limits are checked between requests; an in-flight request is governed by
its configured timeout.

Local `o200k_base` input counts include serialized payload overhead and are
estimates, not billing counts. Reports retain actual provider token usage
separately. Cache hits consume logical inference/token budgets equally; actual
paid requests and cache hits are separate metrics. A shared cache freezes each
state/target policy probability vector across trials while RNG seeds independently
sample actions. API model aliases may change upstream; cached responses record
the returned model identifier. Use a pinned model identifier when available.

### Local query cache and restoring experiments

The default shared cache is `analysis/luna_icl/cache/` (`cache_dir` in
`configs/icl/experiments/base.json`); the demo notebook uses the same directory.
The entire endpoint and request payload, including the model, messages, sampling
settings and output options, determine the key. A change to any of these fields
creates a separate entry. JSON object key order does not matter. Successful
Responses and Embeddings records contain the complete request, the full SDK
response in `response`, and the normalized fields used by the solver. Each file
is named by a SHA-256 digest, so looking up a request does not scan the cache.
The cache has no expiration, entry-count limit, size limit or automatic eviction.

Files are written atomically and a per-request lock prevents concurrent workers
sharing a cache from issuing the same request twice. A failed or incomplete
attempt is not replayed as a success and may be retried. Old cache files remain
usable; entries created before full-response recording retain only the fields
they originally saved. Live responses and mock data live under separate
`openai/` and `mock/` subdirectories. `--replay` reads live entries only,
without an API key or a network request, and reports a cache miss rather than
calling the API. `--execute` reuses matching entries and sends only misses.

To back up and restore the complete cache, run these from the CaRL checkout
when experiment workers are stopped. The destination of each copy must be a
new path:

```bash
cp -a analysis/luna_icl/cache /path/to/backup/luna-query-cache
cp -a /path/to/backup/luna-query-cache analysis/luna_icl/restored-cache
uv run --extra icl python scripts/run_icl_search.py run \
  --config configs/icl/experiments/tuned_fixed.json \
  --cache-dir analysis/luna_icl/restored-cache --replay
```

The restored directory can also be selected in the demo notebook via
`CACHE_DIR`, or passed to `scripts/run_budget_sweep.py` using the same
`--cache-dir` option. Changing a sweep's cache path changes its configuration,
so an existing sweep resume requires its original cache path. To reuse entries
from the previous notebook-specific cache without
overwriting existing shared entries, run
`cp -an analysis/luna_icl/notebook_cache/. analysis/luna_icl/cache/` after
creating the shared directory. Cache files contain full prompts and model
outputs; keep backups wherever those experiment data should be stored.

The cache itself is unlimited; the per-run `limits` still apply to logical
queries, including hits, to keep experiment budgets comparable.

Live/replay and mock caches and calibration identities are separate. Calibration
fingerprints include the model, policy prompt/examples, action sampling settings,
training data and execution cap. Changing them requires a new calibration.
Completed trials are saved atomically; rerunning `calibrate` resumes missing
trials or reuses completed labels. Interrupted in-flight trials may need replay;
invocation costs and attempted actions are recorded separately.

Each run writes resolved config, per-instance action paths and proposal/hit events,
API diagnostics, `results.json`, `results.csv` and `results.md`. Solved plans are
independently replayed before acceptance. Matrix reports add paired success
differences and a board-cluster bootstrap interval, keeping repeated seeds paired.
Few boards make these intervals exploratory. API errors/budget stops are visible
in the reports rather than silently removed. Calibration has separate JSON/CSV/MD.

## Checks and legacy code

For offline Plotly evaluation and an editable notebook, see
[evaluation instructions](configs/icl/evaluation/README.md). `make icl-eval-demo`
recreates short mock runs at 12×12 and 32×32 and exports the interactive dashboard,
tidy CSV/JSON tables and editable `graph_objects` figures. To evaluate live results,
use `python -m scripts.evaluate_icl --runs <matrix-directory>` through `uv run --extra icl`.

```bash
make ruff
make pyright
make pylint-icl
make test
```

Offline verification artifacts are in `analysis/luna_icl/verification/`.
The earlier hybrid runner is retained as `scripts.run_hybrid_icl_search` with
`configs/icl/sokoban_search.json`; its local policy/value checkpoints are **not**
used by the new runner or any experiment matrix. Its historical documentation
is [ICL_HYBRID_LEGACY.md](ICL_HYBRID_LEGACY.md).

API references: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
[Embeddings](https://developers.openai.com/api/docs/guides/embeddings).

### Offline token planning and explicit boards

Use `scripts/estimate_icl_tokens.py` with the quick, pilot or transfer sweep
config to estimate calibration and solver tokens without credentials or API calls.
The commands, editable assumptions, USD price snapshot, caveats and saved
CSV/JSON/Markdown reports are documented in `configs/icl/budget_sweeps/README.md`.
These are workload scenarios, not model measurements or hard billing limits.

The generator is an explicit-state component: it is instructed to return each
complete board as exactly H strings of W ASCII tiles in JSON. Locally, every
proposed board is decoded and checked for the original dimensions, walls, goals,
box count and exactly one player. A latent vector or coordinate-only response is
not accepted. Structural validity does not establish reachability; the actual
CaRL controller must reach the board to accept a subgoal. Changed inference code
invalidates an existing sweep's code-hash resume guard; use a new output directory
for a new experiment rather than mixing old and new generator prompts.

### Inspectable generator checkpoint

The small live study is documented in
[the component report](analysis/luna_icl/campaign_100/generator_probe_v2/report/README.md).
The measured setup uses
[this experiment config](configs/icl/experiments/luna_checkpoint26_retry2.json),
[the complete-board prompt](configs/icl/prompts/generator_planned_board.json), and
[26 saved demonstrations](analysis/luna_icl/campaign_100/generator_probe_v2/checkpoint_balanced/README.md).
The checkpoint includes explicit source/target boards, calibrated labels, the
ordered examples, policy identity, model settings and evidence hashes.
`generator_example_limit` controls the generator context independently of the
CLLP few-shot count. `generator_max_attempts=2` permits one retry with the
static rejection reason; unchanged boards never become accepted subgoals.

Open [the checkpoint inspection notebook](notebooks/luna_checkpoint_inspection.ipynb)
to inspect full saved messages and results without credentials or API access.
The notebook was executed in a fresh kernel with external connections and API
construction denied. Regenerate tables and Plotly figures offline with:

```bash
uv run --extra icl python -m scripts.generator_probe.evaluate
```

Both 4- and 26-example contexts reached 23/30 validation endpoints on their
first generation attempt. A post-hoc retry recovered the seven rejected
26-example cases using replayed first replies. This establishes a useful
component configuration on the observed states, not 30 solved boards or
reliable temporal-scale control. The separate full-solver result and its
trace live under `generator_probe_v2/solver_validation/`.
