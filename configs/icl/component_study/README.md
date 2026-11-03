# Luna component study on 100 held-out trajectories

This study compares **0, 2, and 4 actual input/answer demonstrations** on the
same 100 Sokoban 12×12×4 trajectories. The default config uses `gpt-6-luna`
and the existing tuned prompts/settings; it selects exactly one value example
per training context. The data in `data/icl/component_study_100_clean` were
prepared from the existing offline shard, with distinct training/validation/test
layouts and the three prior holdouts excluded. Witnesses are used for evaluation,
not passed to Luna as query answers.

- **CLLP:** choose a midpoint on each witness trajectory and its state `k=4`
  actions later; report exact-target reach rate within four sampled actions.
- **Generator + CLLP:** from that same midpoint, ask for one explicit `k=4`
  board. Report structural board validity, the reached rate over all evaluable
  generator cases, and reached rate conditional on valid generated boards.
  These are measured with the specified CLLP and budget: failing to reach a
  board does not prove that no policy could reach it. As in the tuned
  configuration, the generator can make up to three attempts with feedback
  when a proposal is rejected; that feedback is not a training demonstration.
- **Value:** evaluate five states at 0%, 25%, 50%, 75%, and 100% progress
  along each witness. The model estimates remaining actions, converted to
  discounted value with the configured discount. Report the fraction of
  non-decreasing adjacent pairs, the fraction of wholly non-decreasing
  trajectories, strict increases and ties. Constant predictions therefore
  cannot appear informative solely by receiving a perfect weak-monotonicity
  score. Witnesses are not assumed optimal.

Every variant uses the same cases and paired, deterministic action-sampling
seeds. Rate intervals are 95% Wilson intervals; mean adjacent-pair metrics use
95% bootstrap intervals resampling whole trajectories, not correlated edges.
Each case uses one controller rollout; intervals describe variation across
these held-out boards, not variability across repeated model samples. The local
query cache freezes previously returned model outputs, and an uncached provider
call can return a different output even with the same input.
Errors and missing cases are visible separately and excluded from rate
denominators. Plots display the number observed and the number planned.

## Plan and run

From the CaRL checkout:

```bash
# Free: show the configured request and token ceilings.
uv run --extra icl python scripts/evaluate_component_study.py --plan

# Paid for local-cache misses. Save each case as soon as it finishes.
uv run --extra icl python scripts/evaluate_component_study.py --execute

# Continue after interruption. Retry recorded errors only when requested.
uv run --extra icl python scripts/evaluate_component_study.py --execute --resume
uv run --extra icl python scripts/evaluate_component_study.py --execute --resume --retry-errors

# Export interactive HTML, summary JSON/CSV/Markdown and usage CSV.
uv run --extra icl python scripts/plot_component_study.py
```

The default upper bound is **4,800 logical calls**, **96 million estimated input tokens**
and **8,793,600 output tokens**; these are conservative ceilings, not expected
consumption or a cost quote. Early controller termination and valid generator
responses reduce logical calls, actual responses can be shorter than token caps,
and cache hits reduce physical API requests. Input limits use a local tokenizer
estimate; provider-reported input usage can differ. `plots/usage.csv` reports physical
requests, input/output tokens, provider prompt-cache token counts, local cache hits and logical requests. Multiply actual
token counts by the pricing applicable to your account to calculate cost.
Each case has its own request, token and time limits in the study config.
Usage covers saved cases and recorded retries. If the process is killed during
a case, requests already made in that unfinished case may be absent from the
report; reconcile final billing with the provider's usage dashboard.

A smaller paid pilot can be started by choosing a component and few-shot counts:

```bash
uv run --extra icl python scripts/evaluate_component_study.py \
  --execute --component policy --shots 0
# Append the other conditions with the same output directory:
uv run --extra icl python scripts/evaluate_component_study.py \
  --execute --component generator --shots 0 2 4 --resume
```

To limit the number of boards for an initial pilot, copy the study JSON, change
`instances` and `output_dir`, and pass `--config path/to/pilot.json`. Keep mock,
replay and live studies in distinct output directories. `--mock` tests plumbing
only; `--replay` never calls the API and only works for cached live requests.
If a case fails, the runner records the error and returns a nonzero exit status;
`--resume --retry-errors` retries only failed records, preserving their earlier
usage. The manifest checks config, provider, data, relevant code and package
versions on resume. If these change, use a fresh output directory; the separate
query cache can still reuse identical requests.

Open `notebooks/luna_component_study.ipynb` to inspect results interactively.
It only reads saved data; it does not start a run. The report and plots can be
exported for a partially completed study, with missing cases clearly shown.

## Recreate the dataset on another checkout

First prepare the normal training/validation/test data and trajectory examples
according to the main ICL documentation. Then run the following. Omit any
`--exclude` path that is not present on that checkout; include any prior holdout
whose layouts should not be reused.

```bash
uv run python scripts/datasets/prepare_fresh_trajectory_holdout.py \
  --config configs/icl/experiments/tuned_fixed.json \
  --output data/icl/component_study_100_clean --count 100 --seed 20261102 \
  --exclude data/icl/fresh_holdout_20261102 \
  --exclude data/icl/fresh_verified_holdout_20261102 \
  --exclude data/icl/final_simple_holdout_20261102
```

The script refuses to overwrite a non-empty dataset directory, verifies every
witness, and writes a manifest containing the source hash and selected IDs.
