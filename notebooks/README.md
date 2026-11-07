# Notebooks

Open the repository's Python 3.11.4 environment after `uv sync --frozen --extra icl`.
Source notebooks are committed without execution outputs. Locally executed
copies from the component studies are retained under
`analysis/notebook_executions/before_source_cleanup_20261103/`.

| Notebook | Purpose |
| --- | --- |
| `luna_goal_progress.ipynb` | Latest saved goal-placement and verbal-judge measurements; offline |
| `luna_checkpoint_inspection.ipynb` | Inspect calibrated demonstrations, payloads and saved solver traces; offline |
| `luna_components_demo.ipynb` | Interactive component and solver demonstration; inspect provider settings before execution |
| `luna_icl_evaluation.ipynb` | Figures from exported live CSV observations; offline |
| `luna_component_study.ipynb` | Component-study tables and figures |
| `luna_trajectory_diagnostics.ipynb` | Trajectory policy, generator and value diagnostics |
| `luna_icl_evaluation_mock_legacy.ipynb` | Separate mock demonstration; not a measured model result |

Saved-result notebooks require the corresponding local artifacts listed in
[`analysis/README.md`](../analysis/README.md). A fresh clone contains source and
launch inputs, not the private cache or all historical results. Missing local
results must not be replaced with mock observations in a live-data report.

## Reachability experiments

`figure1_reachability.ipynb` walks through the reachability measurement in
mock mode. `icl_8v32_reachability.ipynb` and
`icl_proposal_hard_reachability.ipynb` inspect saved live results.
`torch_cllp32_three_families.ipynb` displays local checkpoint results.

The saved results, trajectory witnesses and model checkpoints are not included
in Git. These notebooks need their matching local artifacts and report missing
inputs rather than substituting mock results for measurements.
