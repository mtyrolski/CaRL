# Experiment commands

Run modules from the repository root with `uv run --extra icl python -m`.

| Task | Entry point |
| --- | --- |
| Prepare, preview or run in-context search | `scripts.run_icl_search` |
| Freeze, preview or run checked predicate prompting | `scripts.run_predicate_icl` |
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

## Monitoring local experiments

Run `make show_active_experiments` to see supervised runs, progress, recorded
results and report paths. It only reads run manifests under `analysis/` and
verifies visible process identities; it does not discover arbitrary processes.
Use `watch -n 10 make show_active_experiments` for a refreshing view, or
`make show_active_experiments EXPERIMENT_ARGS=--all` to include completed and
unverified manifests. For machine-readable output, pass `EXPERIMENT_ARGS=--json`.
