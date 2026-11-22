# Experiment commands

Run modules from the repository root using the locked CPython 3.11.4
environment from the [offline quick start](../ICL_SEARCH.md#offline-quick-start).

| Task | Entry point |
| --- | --- |
| Prepare the fixture and run a paired Sokoban sweep | `scripts.run_icl_smoke` |
| Prepare, preview or run in-context search | `scripts.run_icl_search` |
| Freeze, preview or run checked predicate prompting | `scripts.run_predicate_icl` |
| Fit a value heuristic and compare predicate planners | `scripts.run_predicate_icl fit-value` and `search` |
| Compare four planners on native Craftax CPU | `scripts.run_predicate_icl craftax` |
| Evaluate a local full-state Sokoban controller | `scripts.evaluate_full_state_policy` |
| Reproduce the full-state reachability pilot | `scripts.run_full_state_reachability` |
| Plan or run action-budget sweeps | `scripts.run_budget_sweep` |
| Evaluate saved runs offline | `scripts.evaluate_icl` |
| Estimate model token usage | `scripts.estimate_icl_tokens` |
| Prepare or run hybrid full-state search | `scripts.run_hybrid_icl_search` |

Pass `--help` to each command for its options. Configurations are under
`configs/icl/`; datasets, checkpoints, caches and results are local inputs.
The `--mock` and `--replay` modes do not submit live inference requests.
Use the [pinned campaign snapshot](../configs/icl/campaign_data/README.md)
to recover retired configurations, diagnostic scripts, and example notebooks.
To resume a budget sweep, use its exact original source revision, environment
and input paths. Record the commit beside the results. The
`sweep_manifest.json` compares the hashes of the ICL source files and inputs
along with the resolved settings and dependency versions.

## Quick-start profiles

The fixture behind [the Sokoban sweep](../configs/icl/quickstart/sokoban_sweep.json)
provides checked annotations, a support manifest, and validation and held-out
queries. Run the following in the same shell as the quick start. Use a separate
output directory for each profile.

```bash
for profile in zero-shot four-shot openai ministral legacy; do
  uv run --locked --offline --no-sync python -m scripts.run_icl_smoke \
    --config configs/icl/quickstart/sokoban_sweep.json \
    --profile "$profile" --prepare-only --output "$work/$profile"
done
```

The zero-shot and four-shot profiles use the [zero-shot config](../configs/icl/experiments/predicate_zero_shot.example.json)
and [four-shot config](../configs/icl/experiments/predicate_four_shot.example.json).
`profile/preview.json` shows the rendered messages and the number of
examples in each role. `profile/validation.json` and `profile/held-out.json`
record checked mock runs. No model weights or credentials are required.

### Provider profiles

[OpenAI Responses](../configs/icl/experiments/provider_openai.example.json)
sets Responses roles. [Ministral Chat Completions](../configs/icl/experiments/provider_ministral.example.json)
overrides the generator and inherits the other roles. The legacy
[fixed-offset config](../configs/icl/experiments/smoke_fixed.json) supplies one
fallback model. Verify the original CLI, request routing and cache replay
with an injected HTTP transport.

```bash
uv run --locked --offline --no-sync python -m pytest -q -p no:cacheprovider \
  --basetemp="$work/provider-tests" tests/inference_components/test_icl_provider_routes.py
```

For a user-provided experiment, `scripts.run_icl_search preview --config CONFIG.json`
checks the prepared data and renders requests, `run --mock` uses local
components, and `run --replay` uses an existing response cache. `run --execute`
explicitly enables live inference. Supply matching prepared inputs, configured
model IDs and `OPENAI_API_KEY` and/or `MISTRAL_API_KEY` according to the role
routes. Copy [the budget template](../configs/icl/campaign_budget.example.json)
to the ignored `configs/icl/campaign_budget.json` and fill in the campaign
ceiling, stage limits and rates for the selected routes. Reuse one ledger
and keep its config byte-identical across the campaign. The template's zero
ceiling blocks live spending until configured.

## Matched evaluation

The quick-start sweep writes the resolved inputs and per-trial records to its
output directory. Run `scripts.evaluate_icl --matched "$work/sokoban/sweep.json"
--output "$work/report"` to validate the records and compare modes within
identical budgets. Use a new sweep output directory for each invocation.
`REPORT.md`, `evaluation.json`, `comparisons.csv`, `coverage.csv` and
`trials.csv` preserve the estimates and their denominators.

The evaluator reports complete pairs and missing observations separately.
Paired success intervals resample layouts or task families, keeping repeats
together and weighting each unit equally. The restricted time comparison
uses first success when recorded, or elapsed time for a solved trial, capped
at the common wall horizon. An observed run ending without success receives
the full horizon and retains its actual stop time. Missing trials are shown
separately and excluded from paired estimates. Offline compute and adaptation
timings are separate nullable fields.

The full-state budget runner exposes its existing plan separately. The
bundled config resolves without any generated datasets.

```bash
uv run --locked --offline --no-sync python -m scripts.run_budget_sweep \
  --config configs/icl/budget_sweeps/mock.json --plan --output "$work/legacy-sweep-plan"
```

## Checked predicate prompting

`prepare_intent_annotations` writes `D_A.jsonl` and `annotation_audit.jsonl`.
Set `task_group` on source trajectories to preserve groups across different
layouts. The annotation CLI checks group separation before annotating.
`run_predicate_icl freeze` selects examples using a checked `validation_id`
and binds the config and training input to a support manifest. Replace the
example `validation_id` with a record ID from your validation split. Policy
examples use simulator-checked actions from the annotation audit. For scaled
generator examples pass checked `D_C.jsonl` from `measure_intent_policy` and
the separate policy source with `--policy-data`.

`run_predicate_icl preview` takes `--query QUERY.json` and `--intent INTENT.json`
and shows the messages and context use for both roles without credentials.
The query contains `observation: {"rows": [...]}`, `task_goal`, `scale` and the
`split`. A test query also includes `task_group`, using `source_id` where the
dataset has no explicit task group. Group and split are used for isolation
checks. `run` proposes its own intent and executes the policy in Sokoban.
Save a successful validation run with `--output` and pass it as
`--validation-result` on the test split. Select `--mode mock` with fixture text
in `--mock-response`, `--mode replay` with an existing cache, or `--mode openai`
to use the configured live endpoint.

## Planning with checked intents

Use `scripts.run_predicate_icl search --planning-mode MODE` with the frozen
support, query and execution mode used by `run`. Save one successful
validation search per planning mode and pass its output as
`--validation-result` on the test split. The comparison uses one policy and
one action budget. `predicate_sokoban_search.example.json` shows the
`predicate_icl.planning` settings.

- `direct` asks π to achieve the task goal and uses neither G nor V.
- `one-step` tries the configured candidates from the current snapshot and
  memory, scores checked trials with discounted task reward plus the
  nonterminal value, and executes the best intent afresh. The maximum local
  work per decision is scales × `max_proposals` × `trials_per_proposal`
  trial rollouts and one committed rollout, each limited by `rollout_cap`.
- `full-search` pops `(node, scale)` pairs in decreasing order of scale and
  V. An insertion counter resolves ties. A child carries a simulator snapshot,
  action memory and a parent edge. The result reconstructs `action_path`
  from those edges and records rejected trials separately.

Run `scripts.run_predicate_icl fit-value --source DEMOS.json --output V.json
--discount 0.99` to fit V from replay-verified solved training demonstrations.
Set `predicate_icl.planning.value_checkpoint` and use the same `discount`
in the experiment config. Without a fitted checkpoint, the two planning
modes use `roles.value` and the configured API backend. Mock input supplies
a `value` string such as `{"value":0.5}` for that backend.

The JSON report separates `attempted_simulator_actions`, `executed_plan_actions`,
queue expansions, logical calls, physical requests, API and local tokens,
and elapsed time. Failed candidates and no-ops contribute to attempted actions.

## Craftax planning pilot

The symbolic CPU adapter uses Craftax 1.1.0 and JAX 0.4.23. Install the
separate pins into an optional CPython 3.11.4 environment and expose its
`site-packages` to the locked CaRL environment. Set `craftax_wheels` to
a local wheelhouse for an offline install. If the pinned runtime is already
installed, point `PYTHONPATH` in the last command at its `site-packages`.

```bash
uv venv --offline --python 3.11.4 "$work/craftax-venv"
uv pip install --offline --find-links "$craftax_wheels" --no-deps \
  --python "$work/craftax-venv/bin/python" -r requirements/craftax-1.1.0.txt
PYTHONPATH="$work/craftax-venv/lib/python3.11/site-packages" JAX_PLATFORMS=cpu \
  uv run --locked --offline --no-sync python -m scripts.run_predicate_icl \
  craftax --mode mock --planning-mode all --task wood-to-stone-pickaxe \
  --output "$work/craftax/matched.json"
```

All four modes start from the same seeded, staged native world and use the
same visible recipe policy and budgets. The setup records one shared NOOP
from the immutable root to warm up JAX. Each mode starts from the original
root afterwards. Reports include native rewards along trials and endpoint
fingerprints for replay. `plan-and-act-on-failure` acts on an explicit plan
and charges replanning to the generator budget. The task split, observation
contract and comparison with the published method are in [the task
manifest documentation](../configs/icl/craftax/README.md).

The isolated `carl.icl.transfer.discobench` entry point publishes public
benchmark pins and a synthetic transfer contract smoke with its own
provenance. Its public runtime uses a separate Python environment.

## Local checkpoints and LoRA

Install the locked `icl` and `lora` extras when training or loading a PEFT
adapter. Inspecting checked data and running the API-backed profile use `icl`.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for local runs. Supply
the local Hugging Face snapshot path and its exact 40-character revision.
For a snapshot copied out of the Hugging Face cache, put that revision in
`experiment_revision.json` beside the model and tokenizer files.

### LoRA train-and-load smoke

This smoke creates a tiny Qwen3 checkpoint, tokenizer and checked annotations
locally. It performs one PEFT optimizer step and evaluates the exact
validation-selected adapter on the held-out split. The smoke fixture is
generated by the repository's round-trip test.

```bash
uv sync --locked --offline --no-dev --python 3.11.4 --extra icl --extra lora
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
lora_work="$work/lora"
uv run --locked --offline --no-sync python - "$lora_work/fixture" <<'PY'
from pathlib import Path
import runpy
import sys

fixture = runpy.run_path("tests/experiments/test_local_lora_roundtrip.py")
fixture["build_fixture"](Path(sys.argv[1]))
PY
base="$lora_work/fixture/base"
revision="$(uv run --locked --offline --no-sync python -c \
  'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["revision"])' \
  "$base/experiment_revision.json")"
runtime=(--model "$base" --revision "$revision" --device cpu --dtype float32
         --threads 1 --max-length 512 --max-new-tokens 2)
da="$lora_work/fixture/annotations/D_A.jsonl"
uv run --locked --offline --no-sync python -m carl.icl.intent_generator_experiment inspect \
  --data "$da" --output "$lora_work/inspect.json"
uv run --locked --offline --no-sync python -m carl.icl.intent_generator_experiment train \
  --data "$da" --output "$lora_work/train" "${runtime[@]}" \
  --max-steps 1 --eval-every 1 --accumulation 1 --rank 2 \
  --lora-target q_proj --lora-target v_proj --pad-to-length 224
adapter="$(uv run --locked --offline --no-sync python -c \
  'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["adapter"])' \
  "$lora_work/train/best.json")"
uv run --locked --offline --no-sync python -m carl.icl.intent_generator_experiment evaluate \
  --data "$da" --output "$lora_work/evaluated.jsonl" "${runtime[@]}" --adapter "$adapter"
```

### Training with checked local data

- Inspect and train on checked demonstrations with
  `python -m carl.icl.intent_generator_experiment {inspect,train} --data D_A.jsonl`.
  The demonstration prompt uses a null scale.
- Inspect and train on controller measurements with
  `python -m carl.icl.policy_relative_lora {inspect,train} --data D_C.jsonl --arm scaled`.
  Use distinct successful controller durations in train and validation for a
  scaled experiment. Run the data and resource checks before increasing the
  training budget.
- Both `train` commands accept explicit `--model`, `--revision`, `--output`,
  `--device`, `--dtype`, `--rank`, repeatable `--lora-target`, `--lr`,
  `--seed`, `--max-steps`, `--accumulation` and `--eval-every`. Pass the exact
  validation-selected checkpoint directory named in `best.json` to the
  corresponding `evaluate` command as `--adapter`. Set `--requested-scale`
  for scaled evaluation. The checkpoint directory contains the safetensors
  adapter, its JSON config, `tokenizer/`, and `adapter_manifest.json`.
- Add `predicate_icl.local_roles.generator` or `.policy` to the ICL config
  to use a local role. See `predicate_local_generator.example.json` and set
  your verified snapshot, selected checkpoint and validation record ID.
  Use `--mode local` when both roles are local. For a mixed run select the
  execution mode of the API role. In mock mode provide responses only for
  API roles. `measure_intent_policy --local CONFIG --support-manifest FILE`
  uses the frozen local π and checks that the measurement pool is held out.

## Full-state controller pilots

`python -m carl.run --config-dir "$PWD/configs/icl/full_state" --config-name
sokoban_bert_cllp_pilot` trains a tiny BERT CLLP from scratch for one step
using the configured local 12×12 Sokoban trajectories. The default Hydra
run directory is `analysis/icl/full_state/bert`. The evaluator config
`sokoban_bert_eval.example.json` points to that run's `checkpoint-1`.

`sokoban_qwen3_lora_eval.example.json` pins the existing Qwen3-0.6B
snapshot and the rank-eight coordinate-policy adapter by its file hashes.
Provide that local snapshot and the adapter's `adapter_config.json` and
`adapter_model.safetensors`. The Qwen controller ranks four direction tokens
from the current and target boards with player coordinates. Set `device`
and `dtype` for the local hardware.

Evaluate either full-state policy with `python -m scripts.evaluate_full_state_policy
--config CONFIG.json --pair PAIR.json --output RESULT.json`. The pair contains
the `source` and `target` board rows and an `action_cap`. Results identify
the pinned weights and report the actions and exact reachability.

## Full-state reachability (Figure 1)

```bash
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_smoke.json --output "$work/figure1"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_campaign.json \
  --plan-only > "$work/figure1-campaign-plan.json"
```

The [smoke config](../configs/icl/full_state/reachability_smoke.json) generates
a small local witness fixture and runs the mock controller. The
[campaign manifest](../configs/icl/full_state/reachability_campaign.json)
specifies support or training trajectories on 12×12 four-box layouts and
disjoint test trajectories on 12×12 ten-box layouts. Each test request uses
the current board and a future target board from the same verified trajectory.
The measured event is the first exact target hit within `k` attempted actions.
The output includes `report.json`, `points.csv` and `reachability.svg` with
valid case counts, coverage and pointwise 95% Wilson intervals. Full points
target 100 valid trials and reserve 20 further distinct layouts per offset.
Models can declare their own offset subsets.

The manifest names the two trajectory inputs. Each uses `schema_version: 1`,
a `split` of `train` or `test`, and `trajectories` with an `id`, twelve
`board` rows and a solved `actions` witness encoded as integers from 0 to 3.
The smoke writes the same input format as `train-fixture.json` and
`test-fixture.json`. Supply the BERT checkpoint and the Qwen3 snapshot with
its pinned adapter as described under [full-state controller pilots](#full-state-controller-pilots).
[Detailed witness requirements](../configs/icl/full_state/REACHABILITY.md).

The offline plan lists capped requests and tokens. For USD estimates add
`--prices TARIFFS.json` with an object keyed by condition ID, each containing
`input_usd` and `output_usd` per million tokens. Live billing uses the
campaign ledger. To launch a selected API condition, retain the full
manifest, supply its prepared trajectory files and compatible credentials,
and run with `--execute-live --model CONDITION_ID --output DIRECTORY`.
Rerunning an unchanged manifest in that result directory resumes saved
trials. `--report-only --output DIRECTORY` rebuilds the plots from records
without inference.

## Monitoring local experiments

Run `make show_active_experiments` to see supervised runs, progress, recorded
results and report paths. It only reads run manifests under `analysis/` and
verifies visible process identities; it does not discover arbitrary processes.
Use `watch -n 10 make show_active_experiments` for a refreshing view, or
`make show_active_experiments EXPERIMENT_ARGS=--all` to include completed and
unverified manifests. For machine-readable output, pass `EXPERIMENT_ARGS=--json`.
