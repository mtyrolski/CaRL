# In-context search

CaRL supports exact target-board search and search over checked predicate
intents. An intent specifies effects to reach and conditions to preserve during
execution. Reports count attempted primitives, including no-ops and failed
rollouts, separately from committed actions, model calls, and wall time.

## Offline quick start

Run from the repository root with locked CPython 3.11.4. Add `--offline` to
`uv sync` when Python and the locked distributions are on disk.

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

The bundled mock prepares annotations and frozen support, runs three planning
modes on matched boards and budgets, and writes per-trial records. The offline
report shows paired differences, missing pairs, coverage, and costs. Intervals
keep repeats from a layout or family together. For a prompt preview and
validation fixtures, add `--prepare-only --profile NAME` and use a new output
directory. The available profiles are `zero-shot`, `four-shot`, `openai`,
`ministral`, and `legacy`.

Inspect the full-state budget sweep with this offline plan command. Add
`--mock` for fixture execution, `--replay` for saved responses, or
`--execute` for live inference. Pass `--resume` with the same configuration
and inputs.

```bash
uv run --locked --offline --no-sync python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/mock.json --plan --output "$work/sweep-plan"
```

## Models and providers

Run `scripts.run_icl_search preview --config CONFIG.json` through `uv run` to
render prepared messages. Its `run` command accepts `--mock`, `--replay`, or
`--execute`. Inherited `model` covers roles without an override. Set
`roles.policy`, `roles.generator`, and `roles.value` to pin model, provider,
API, revision, and decoding independently. A changed controller requires
matching calibration. [OpenAI Responses example](configs/icl/experiments/provider_openai.example.json)
· [Ministral Chat Completions example](configs/icl/experiments/provider_ministral.example.json)

Live execution reads `OPENAI_API_KEY` or `MISTRAL_API_KEY` according to the
configured routes. Copy the budget template to the ignored
`configs/icl/campaign_budget.json`, allocate a stage, and set
`CARL_CAMPAIGN_STAGE` to its ID. Price routes by `provider/endpoint/model`
with `input_usd_per_million`, `max_input_tokens`, and the corresponding output
fields for generation endpoints. The example has a USD 0 ceiling.
Reuse the same budget configuration and ledger throughout the campaign.
[Budget template](configs/icl/campaign_budget.example.json)

Use `scripts.estimate_icl_tokens --help` for offline token estimates and
`scripts.run_hybrid_icl_search --help` for a generator with local policy and value.

## Checked predicate planning

Use `scripts.prepare_intent_annotations --source DEMOS.json --output DIR`
to build checked `D_A.jsonl` and `annotation_audit.jsonl`. Use
`scripts.measure_intent_policy --data D_A.jsonl --output DIR` to build
`D_C.jsonl` from directed controller attempts. Select `--mock`, `--replay`,
or `--local` as appropriate. Freeze groups before preparing data and keep
source families and symmetric layouts apart across splits.

`scripts.run_predicate_icl freeze --config CONFIG.json --data D_A.jsonl --manifest SUPPORT.json`
binds training support to a real validation record ID. For a scaled generator,
pass checked `D_C.jsonl` as `--data` and the policy's `D_A.jsonl` as
`--policy-data`. The `preview` command adds `--query QUERY.json --intent INTENT.json`.
A query contains visible `observation.rows`, `task_goal`, `scale`, and `split`.
Test queries also include `task_group`.

Use `run` for one proposed intent or `search --planning-mode` for
`direct`, `one-step`, or `full-search`. Direct control targets the task.
One-step probes candidates from the current snapshot and commits the chosen
intent afresh. Full search prioritizes pairs of node and scale by decreasing
scale and value. Save a successful validation output for each condition and
pass it with `--validation-result` on test. Use `--mode mock --mock-response FILE`
for fixtures. Fit a value checkpoint with `fit-value --source DEMOS.json --output V.json --discount 0.99`
and use the same discount in planning. [Search configuration](configs/icl/experiments/predicate_sokoban_search.example.json)

## Craftax on CPU

The optional adapter uses Craftax 1.1.0 and JAX 0.4.23. Set `craftax_wheels`
to a cached wheelhouse to install the pinned runtime offline.

```bash
uv venv --offline --python 3.11.4 "$work/craftax-venv"
uv pip install --offline --find-links "$craftax_wheels" --no-deps \
  --python "$work/craftax-venv/bin/python" -r requirements/craftax-1.1.0.txt
PYTHONPATH="$work/craftax-venv/lib/python3.11/site-packages" JAX_PLATFORMS=cpu \
  uv run --locked --offline --no-sync python -m scripts.run_predicate_icl \
  craftax --mode mock --planning-mode all --task wood-to-stone-pickaxe \
  --output "$work/craftax/matched.json"
```

The four modes share a staged, seeded native start. The fixture uses a recipe
policy. Plan-and-Act-on-failure requests a typed plan and charges replanning
to the shared call budget when a safe attempt fails. The model sees visible
observations and action memory. Native inventory checks verify effects and
preservation. [Task manifest](configs/icl/craftax/tasks.json)
· [Plan-and-Act paper](https://proceedings.mlr.press/v267/erdogan25a.html)

With a pinned local controller JSON containing `model_id`, `threads`, and
a `local` snapshot configuration, use `--mode local --planning-mode direct
--local-controller CONFIG.json`. Keep the Craftax runtime on `PYTHONPATH`
and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for cached weights.
`carl.icl.transfer.discobench --help` covers the separate transfer fixture.

## Local checkpoints and LoRA

Install the locked `icl` and `lora` extras to train or load a PEFT adapter.
Supply the local snapshot and its exact revision. Put that revision in
`experiment_revision.json` for snapshots copied out of the Hugging Face cache.

Use `python -m carl.icl.intent_generator_experiment {inspect,train,evaluate}`
with checked `D_A.jsonl`. For measured scale use
`python -m carl.icl.policy_relative_lora {inspect,train,evaluate} --arm scaled`
with checked `D_C.jsonl` containing informative train and validation contexts.
For training, pass `--data`, `--output`, `--model`, and `--revision` and set
PEFT and step limits as needed. Pass the checkpoint selected in `best.json`
to `evaluate --adapter`. The CPU round-trip fixture runs through
`python -m pytest tests/experiments/test_local_lora_roundtrip.py`.
[Local generator example](configs/icl/experiments/predicate_local_generator.example.json)

Evaluate a pinned BERT or Qwen3 LoRA full-state controller with
`scripts.evaluate_full_state_policy --config CONFIG.json --pair PAIR.json --output RESULT.json`.
The pair supplies `source` and `target` board rows and an `action_cap`.
[BERT evaluator](configs/icl/full_state/sokoban_bert_eval.example.json)
· [Qwen evaluator](configs/icl/full_state/sokoban_qwen3_lora_eval.example.json)

## Full-state reachability

```bash
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_smoke.json --output "$work/figure1"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_campaign.json --plan-only
```

Figure 1 measures the first exact target-board hit within `k` attempted
actions at demonstration offset `k`. A task solution before that target
counts as a miss. The campaign uses four-box support or training layouts and
disjoint ten-box test layouts, all 12×12. Each full point targets 100 valid
layouts with reserves for technical errors. `points.csv` and the SVG show
denominators and pointwise 95% Wilson intervals.

Supply the train and test JSON files named in the campaign manifest. Each has
`schema_version: 1`, `split`, and a `trajectories` list of unique `id`, twelve
ASCII `board` rows, and solved `actions` witnesses encoded UP, DOWN, LEFT,
RIGHT as 0–3. Use at least four independent train layouts and 120 eligible
test layouts per offset with the configured reserve. The smoke writes files
in this format. Keep witness actions out of controller inputs.
[Campaign manifest](configs/icl/full_state/reachability_campaign.json)

`--prepare-only --output DIRECTORY` freezes inputs without loading weights.
Local conditions need their exact checkpoints and training receipts with
`training_split_fingerprint` from the train manifest and
`policy_fingerprint` from `scripts.evaluate_full_state_policy`. Train the BERT
pilot with `python -m carl.run --config-dir "$PWD/configs/icl/full_state"
--config-name sokoban_bert_cllp_pilot` and supply the configured local trajectories.
[BERT training configuration](configs/icl/full_state/sokoban_bert_cllp_pilot.yaml)

The plan reports request and token ceilings. Pass `--prices TARIFFS.json`
with `input_usd` and `output_usd` per million tokens keyed by condition.
Live API execution requires one `--model CONDITION_ID`, `--execute-live`,
`--output DIRECTORY`, and the configured provider ledger. Use the same
directory to resume unchanged inputs or rebuild plots with `--report-only`.

Keep generated results, data, model weights, credentials, and caches local.
Use `make show_active_experiments` to inspect supervised local jobs.
