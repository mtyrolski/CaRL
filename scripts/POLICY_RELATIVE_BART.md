# Small BART from scratch

Run from `/home/mtyrolski/dev/carl/CaRL`. The generator starts from random weights. Only the frozen controller uses trained weights. Scratch reads `configs/generator/sokoban_bart_small.json`; it does not read the generator checkpoint directory.

```bash
cd /home/mtyrolski/dev/carl/CaRL
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. HF_HUB_OFFLINE=1 \
uv run --frozen python scripts/train_policy_relative_bart.py \
  --initialization scratch \
  --model-config configs/generator/sokoban_bart_small.json \
  --dataset analysis/policy_relative_smoke_20261029/main \
  --conditions relative \
  --steps 300 --batch 8 --lr 3e-4 --eval-every 100 \
  --output "analysis/policy_relative_runs/scratch_$(date +%Y%m%d_%H%M%S)"
```

This is a small pilot on the existing cached 750 examples (96 contexts, 48 layouts, 32/8/8 train/validation/test layouts), not full-data training. The fixed seed comes from the calibration metadata, currently `20261029`. Calibration is reused rather than repeated. `u` is derived from measured frozen-policy execution times and success frequencies, not from the hypothetical figure data.

For the matched comparison, replace `--conditions relative` with `--conditions relative absolute none`. The conditions run sequentially with identical random initialization, samples, model size and update budgets. `--steps` controls the number of updates per condition. More updates on this tiny dataset are not a substitute for more training contexts.

## Files

| Path | Role |
|---|---|
| `scripts/train_policy_relative_bart.py` | BART model, scalar input, training, loss evaluation, autoregressive generation and controller evaluation |
| `configs/generator/sokoban_bart_small.json` | Standalone architecture: 256 width, 3+3 layers, 4 heads, FFN 2048/1024; no weights |
| `scripts/policy_relative_smoke.py` | Existing calibration/split/rollout/validity helpers; also contains the historical CNN/MLP pilot |
| `tests/experiments/test_policy_relative_bart.py` | Checks scratch never loads pretrained generator weights, scalar wiring and 290-token inputs |
| `tests/experiments/test_policy_relative_smoke.py` | Calibration ties, exclusions and board validity checks |
| `analysis/policy_relative_smoke_20261029/main/` | Cached measured labels, split metadata and provenance |
| `analysis/policy_relative_runs/scratch_<timestamp>/` | Your new run's outputs |

The architecture's positional capacity is extended from 256 to 512 to accommodate current board + goal (290 tokens), and the scalar projection adds 256 parameters. The complete generator has 7,375,616 parameters. The scale projection starts at zero; the BART weights are random. `pretrained` remains an explicit opt-in solely for reproducing the historical diagnostics; it is not the agreed training protocol.

## Outputs and scope

The runner prints its initialization and `generator_weights_loaded: false` before training. It writes the exact executed source and updates `learning_curves.csv` at every evaluation. On completion it writes `summary.json`, `metrics.csv`, `generated_boards.npz`, and `<condition>/checkpoint/`, `scale_projection.pt`, `trainer_state.json`. Both the BART checkpoint and scalar projection are needed for conditioned inference. Original data and policy checkpoints are read only.

This is a bounded experiment runner, not yet the final training system: it has no resume/optimizer-state checkpointing, no validation-based best-checkpoint selection, and no multi-GPU or streaming-data support. It currently logs test loss at evaluation intervals; do not use those measurements to select settings. Train/validation/test layouts are separated, but old policy pretraining exposure is unknown. The saved model is the final update, not the best validation checkpoint.

The generator emits explicit boards compatible with the existing controller. The proposal's latent-compatible controller and latent hit criterion remain separate work. Free-generation structural validity and controller reachability matter alongside teacher-forced loss. The existing 300-update scratch pilot reduced loss but generated no structurally valid boards in 32 requests.

To regenerate calibration on a larger bounded sample, inspect `uv run --frozen python scripts/policy_relative_smoke.py --help`. That older entry point also runs the historical small-head matrix after calibration; calibration has not yet been extracted into a dedicated CLI. Keeping that limitation explicit avoids presenting the pilot as a finished production trainer.

## Run all three conditions with persistent logs

```bash
cd /home/mtyrolski/dev/carl/CaRL
bash scripts/run_policy_relative_bart.sh
```

This runs the same 300-update scratch pilot sequentially for `relative`, `absolute`
and `none`. An optional first argument specifies a new output directory. Existing
output directories are rejected. Each run directory contains:

- `console.log`: full stdout/stderr, streamed to the terminal as well;
- `status.json`: `running`, then `completed` or `failed` with the exit code;
- `command.sh`, `runner_used.py`, `model_config.json`, `calibration_metadata.json`: command and provenance;
- `learning_curves.csv`: updated at every evaluation;
- `summary.md`, `loss_changes.csv`: start/end loss and signed changes;
- `learning_curves.html`, `.svg`, `.pdf`, `.png`: train/validation curves;
- the existing model checkpoints, full metrics and generated-board diagnostics.

The reporting code lives in `scripts/summarise_policy_relative_bart.py` and can
be rerun on a completed directory without repeating training. The launch script
is a local logging wrapper, not a resume mechanism.
