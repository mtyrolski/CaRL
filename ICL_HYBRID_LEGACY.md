> Legacy hybrid experiment, retained for reference. The current all-Luna workflow is [ICL_SEARCH.md](ICL_SEARCH.md).

# OpenAI ICL search

Run the existing CaRL search with an OpenAI subgoal generator, frozen local CLLP,
and frozen local value model. The default model is **`gpt-6-luna`**, as requested.
No generator training is needed. The default command prepares requests offline.

## Quick start

```bash
cd /home/mtyrolski/phd/2026/CaRL
uv sync --frozen --extra icl

# Export reproducible inputs and inspect initial prompts; no API calls.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search --prepare-data

# Full solver and real local CLLP/value, with labelled mock proposals.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search --mock --easy --num-subgoals 4

# Set OPENAI_API_KEY in your shell, then run one easy case with Luna.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search \
  --execute --easy --instances 1 --max-requests 3

# Original starting states, adaptive search with k = [8, 4, 1].
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search --execute

# Select four demonstrations from eight layouts by cosine similarity.
# The budget includes both embedding and generation requests.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search \
  --execute --selection embeddings --instances 1 --max-requests 40

# Same generator interface, using the existing universal planner without k.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search \
  --execute --planner universal

# Re-run a prior live configuration using only completed cached responses.
# Use the same planner, selection, instance set and generation parameters.
uv run --frozen --extra icl python -m scripts.run_hybrid_icl_search --replay
```

`--easy` uses states one primitive action from solved. It is a cheap interface
diagnostic, not a planning benchmark. Mock proposals are legal one-step successors
enumerated locally, not model outputs or genuine k-step predictions. Mock/live
caches are separate. `--replay` reads only the live cache and never calls the API.
Replay stops at the first cache miss, including after a previously budgeted run.

Change the model with `--model`, reasoning with `--reasoning-effort`, and prompt
size with `--examples-per-request`. Luna defaults to `reasoning.effort=none` for
the first speed baseline; try `--reasoning-effort low` for comparison. Use
`--reasoning-effort omit` when testing a model without this parameter. Model IDs
are passed through unchanged; there is no automatic model fallback.

## Configuration and inputs

The configuration is [configs/icl/sokoban_search.json](configs/icl/sokoban_search.json).
Paths in that config are relative to the CaRL root, including when the runner is
started from another directory. Local dependencies are installed in this copy's
own `.venv`. The copy's `rl-data`, `models_weights` and `validation` links reuse
existing data and checkpoints under `~/dev/carl`; inference only reads them.
Adjust those links or the JSON paths when moving the checkout.

`--prepare-data` reads a local offline trajectory shard and exports:

- `configs/icl/search_demonstrations.json`: eight distinct layouts, each with
  `k=8,4,1` examples from actual trajectory offsets. Four matching examples are
  used per adaptive request by default. Universal mode ignores the k labels.
- `configs/icl/search_instances.json`: two other layouts at their original starts.
- `configs/icl/search_instances_easy.json`: the same evaluation layouts, one move
  from solved.

Preparation checks every transition in the selected trajectories by replaying it
in the real environment, records the source SHA-256 and seed, and splits by wall
and goal layout. The runner rejects demonstration/evaluation layout overlap.
Evaluation solution states and action sequences are never included in prompts.
These are small within-shard holdouts; previous exposure of the pretrained local
controller or OpenAI model is unknown.

You can supply your own inputs with the same JSON schema:

```json
{"demonstrations": [{"state": ["ASCII rows"], "subgoal": ["ASCII rows"], "k": 8}]}
```

```json
{"instances": [{"id": "my_board", "state": ["ASCII rows"]}]}
```

Tile symbols follow the environment: `#` wall, `-` floor, `.` goal, `$` box,
`*` box on goal, `@` player, `+` player on goal. Boards remain one-hot arrays
inside CaRL. Configure the matching square board size, box count and checkpoints.

## Direct context and embeddings

`direct` takes the first matching demonstrations in the prepared file. Set
`--examples-per-request 8` to include the full default pool for an adaptive
horizon. The examples form a stable prefix followed by the current search state.
For GPT-6 models, an explicit 30-minute cache breakpoint marks that prefix.

`embeddings` embeds the ASCII board plus explicit wall/goal/box/player coordinates
with `text-embedding-3-small`, ranks matching-horizon demonstrations by cosine
similarity, and includes the nearest examples as text. The retrieval query uses
only the current board; reference targets are not embedded for ranking. Each
board embedding is cached independently, so shared demonstration boards across
horizons are reused. Embeddings from this separate model select context; Luna
still receives explicit boards. Spatial usefulness of text embeddings is an
experimental question, not an assumed property.

The first embedding run needs one request per uncached demonstration layout plus
one for the query, before generation. Later search states may need new query
embeddings. This is why the example embedding command has a larger request cap.

## Preserved search behavior and limits

The adapter implements `construct_network()`, `get_network()` and
`get_subgoals(node)`, returning CaRL `GeneratedSubgoal` objects. The runner uses
the existing `Solver`, `AdasubsPlanner`/`UniversalPropositionalPlanner`,
`BasicValidator`, `TransformerConditionalLowLevelPolicy` and `TransformerValue`.
Adaptive k priorities and k-step validation limits are unchanged. Universal mode
uses an exact configurable action cap; the runner accounts for BasicValidator's
historical `budget + 1` fallback. CLLP chooses greedy primitive actions, matching
BasicValidator rather than the sampled controller in the earlier ICL script.

The proposal distribution changes: a JSON list generated by OpenAI replaces
local BART decoding. This does not reproduce BART beam search or its probabilities.
Before execution, proposals must preserve dimensions, walls, goals, box count
and one player. Self-copies and within-response duplicates are filtered. The
local controller then checks actual reachability. Every reported solution is
independently replayed from its starting state.

Defaults: twelve total API attempts per run, two subgoals per expansion, 2,048
maximum output tokens per generation, 100 high-level expansions, CaRL's 500
low-level node budget, 60-second request timeout, and a 300-second search wall
budget checked before API/cache operations. One in-flight request can overrun
the wall budget by its timeout. SDK retries are disabled. Budgets apply across
all evaluation instances in a run; errors consume request attempts, cache hits
do not. Embedding calls share the request budget. There is no account-wide dollar
cap or multi-process budget coordinator. Use one runner process per budget.

## Results and troubleshooting

Each invocation writes its resolved config and outputs to a new directory under
`analysis/icl_search/runs/`. Preparation writes `prepared_requests.json` for initial
nodes only; later requests depend on the evolving search. In embedding mode the
offline preview uses file order and is explicitly labelled as unresolved.

Execution writes `summary.json` after each attempted instance: provider, model,
planner, context selection, solve status, replayed action/subgoal paths, low-level
work, accepted/rejected proposals, per-k reachability, API timings, tokens, cache
hits, and generator format/duplicate/self-copy counts. Budget exhaustion and API
errors are labelled separately from a normally exhausted search. A request error
ends the run with exit code 1 and preserves completed instance results.

Requests and responses are stored under `analysis/icl_search/cache/openai/` or
`mock/`, keyed by the exact endpoint and payload. An incomplete/refused/error
response is not a reusable success. Raw exception messages and credentials are
not saved. The runner reads `OPENAI_API_KEY` from the environment, does not load
the original `.tokens.env`, and sends `store=False` with Responses requests.
Authentication/permission errors stop the run; there is no silent fallback.

Focused tests:

```bash
uv run --frozen --extra icl pytest tests/inference_components/test_openai_icl_search.py tests/solver tests/planners -q
uv run --frozen pyright carl/icl scripts/run_icl_search.py
```

Local verification passed all 15 new integration tests, Pyright and Ruff. Both
adaptive and universal mock runs solved and independently replayed the two easy
cases using the actual frozen CLLP/value checkpoints. No live model call was
made. `make test` reported 132 passed and six pre-existing failures: tests for
five missing legacy scripts (`gather_k841_solutions`, `inspect_k841_trajectories`,
`build_generator_targets_grid`, `estimate_batch_capacity`, and
`compare_trajectory_characteristics`). The same six failures were reproduced in
the original `~/dev/carl/CaRL` checkout.

Official API references: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), and
[embeddings](https://developers.openai.com/api/docs/guides/embeddings).
