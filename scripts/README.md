# Experiment commands

Run modules from the repository root with `uv run --extra icl python -m`.

| Task | Entry point |
| --- | --- |
| Prepare, preview or run in-context search | `scripts.run_icl_search` |
| Freeze, preview or run checked predicate prompting | `scripts.run_predicate_icl` |
| Evaluate a local full-state Sokoban controller | `scripts.evaluate_full_state_policy` |
| Plan or run action-budget sweeps | `scripts.run_budget_sweep` |
| Evaluate saved runs offline | `scripts.evaluate_icl` |
| Estimate model token usage | `scripts.estimate_icl_tokens` |
| Prepare or run hybrid full-state search | `scripts.run_hybrid_icl_search` |

Pass `--help` to each command for its options. Configurations are under
`configs/icl/`; datasets, checkpoints, caches and results are local inputs.
The `--mock` and `--replay` modes do not submit live inference requests.

The structured-intent generator is available as
`python -m carl.icl.intent_generator_experiment --help`.

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

## Local checkpoints and LoRA

Install the locked `icl` and `lora` extras when training or loading a PEFT
adapter. Inspecting checked data and running the API-backed profile use `icl`.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for local runs. Supply
the local Hugging Face snapshot path and its exact 40-character revision.
For a snapshot copied out of the Hugging Face cache, put that revision in
`experiment_revision.json` beside the model and tokenizer files.

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

## Monitoring local experiments

Run `make show_active_experiments` to see supervised runs, progress, recorded
results and report paths. It only reads run manifests under `analysis/` and
verifies visible process identities; it does not discover arbitrary processes.
Use `watch -n 10 make show_active_experiments` for a refreshing view, or
`make show_active_experiments EXPERIMENT_ARGS=--all` to include completed and
unverified manifests. For machine-readable output, pass `EXPERIMENT_ARGS=--json`.
