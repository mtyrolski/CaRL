# Experiment commands

Use `uv run --extra icl python -m ...` from the repository root.

| Task | Entry point |
| --- | --- |
| Plan, launch or resume the current 100-board comparison | `scripts.launch100` |
| Export its CSV and HTML/SVG/PDF figures offline | `scripts.launch100.report` |
| Plan or run the matching local Torch comparison | `scripts.trained_reference.run` |
| Recover the original native k=8,4,1 historical reference | `scripts.trained_reference.freeze` |
| Generic configurable explicit-state solver | `scripts.run_icl_search` |
| Generic independent action-budget sweep | `scripts.run_budget_sweep` |
| Generic saved-run evaluation | `scripts.evaluate_icl` |
| Check notebook types | `scripts.check_notebook_types` |

Start with [`LAUNCH_100.md`](../LAUNCH_100.md). The current launch reads its
versioned public boards and training assets directly. It does not rerun data
generation, calibration or the historical campaign tools.

Supporting code is grouped by purpose:

- `datasets/`: source indexing, deterministic split preparation and verification.
- `training/`: local policy-relative BART experiments.
- `generator_probe/`, `goal_probe/`: focused component studies and their reports.
- `verbal_icl/`: experimental verbal controller, model judge and calibration.
- `experiment_support/`: shared limits, trace recording and replay helpers.
- `archive/campaign_20261102/`: historical campaign-specific orchestration.

The archive README maps old script names to their current modules. There are no
forwarding wrappers at the old locations. Historical run manifests and frozen
source archives retain the code identities that actually produced their data.
