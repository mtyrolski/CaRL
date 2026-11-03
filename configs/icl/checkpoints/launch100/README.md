# Portable 100-board launch inputs

`assets/manifest.json` binds every runtime asset by SHA-256 and records its origin. These files are sufficient for the launch loaders; historical `analysis/`, private witnesses, the source trajectory archive, and the SQLite source index are not runtime dependencies.

The evaluation cohort is the existing **100 real 12×12 boards** from `campaign_dataset_12x12_v1`, selected with seed 20261117. It is not the older mixed 50-real/50-generated cohort. The export checks 100 distinct canonical layouts, no intersection with the ten generator-training layouts or 22 known validation layouts, and every source trajectory index, source digest, initial-state hash and layout ID against the existing metadata index. No held-out solution or action witness was opened. This check does not constitute a new replay of the source trajectories.

- `data/`: original frozen policy training, value examples and public pilot manifest. Public pilot records contain no witness actions or future states.
- `checkpoint/`: unchanged 26-example measured generator checkpoint and its adjacent calibration evidence. Labels are measured frozen-policy labels, not recorded trajectory offsets.
- `demonstrations.json`: six verified training-only task-progress snippets. They have no `u` or `d` labels.
- `config.json`: resolved explicit configuration, portable asset paths, 500 expansion limit and separate 4000-action limit. Launch conditions and limits are controlled by the launch specification.
- `verbal.json`: six measured semantic calibration examples and original controller/judge demonstrations. Only the generator prompt changes to the grounded prompt selected after observed validation failures. This is an experimental carried-forward configuration, not an independently validated improvement.
- `cohort_checks.json`: source identity checks and explicit train/validation/evaluation layout sets.

The training provenance intentionally retains the original absolute source path as archival text: its exact bytes participate in the frozen policy fingerprint. Loaders must preserve it and must not attempt to open that path. The live generator checkpoint path inside the verbal reference config is relocated; its original config and hash remain in provenance.

Export/reverification is offline and requires the original local preparation inputs:

```bash
uv run --extra icl python -m scripts.goal_probe.export_launch_assets
```

The exporter rejects different existing artifacts. Runtime loading verifies included hashes and does not rerun dataset preparation or calibration. New paid execution requires the launch runner's explicit execution command and its budget checks.
