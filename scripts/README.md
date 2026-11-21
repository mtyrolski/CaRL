# Experiment commands

Run modules from the repository root with `uv run --extra icl python -m`.

| Task | Entry point |
| --- | --- |
| Prepare, preview or run in-context search | `scripts.run_icl_search` |
| Plan or run action-budget sweeps | `scripts.run_budget_sweep` |
| Evaluate saved runs offline | `scripts.evaluate_icl` |
| Estimate model token usage | `scripts.estimate_icl_tokens` |
| Prepare or run hybrid full-state search | `scripts.run_hybrid_icl_search` |

Pass `--help` to each command for its options. Configurations are under
`configs/icl/`; datasets, checkpoints, caches and results are local inputs.
The `--mock` and `--replay` modes do not submit live inference requests.

The structured-intent generator is available as
`python -m carl.icl.intent_generator_experiment --help`.

## Monitoring local experiments

Run `make show_active_experiments` to see supervised runs, progress, recorded
results and report paths. It only reads run manifests under `analysis/` and
verifies visible process identities; it does not discover arbitrary processes.
Use `watch -n 10 make show_active_experiments` for a refreshing view, or
`make show_active_experiments EXPERIMENT_ARGS=--all` to include completed and
unverified manifests. For machine-readable output, pass `EXPERIMENT_ARGS=--json`.
