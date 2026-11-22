# Figure 1 protocol and campaign inputs

Run these commands from the repository root with a lockfile-installed CaRL
`icl` environment. The campaign config measures transfer from 12×12 Sokoban
with four boxes to disjoint 12×12 layouts with ten boxes. Each point uses `k`
as both the demonstration offset and the controller's maximum actions. The
metric is first exact board equality, including the player and every box.
A task terminal state reached before the target is a valid miss.

## Verify the protocol offline

```bash
mkdir -p analysis/icl
work="$(mktemp -d "$PWD/analysis/icl/reachability.XXXXXX")"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_smoke.json --output "$work/smoke"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --report-only --output "$work/smoke"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config configs/icl/full_state/reachability_campaign.json --plan-only
```

The smoke generates verified local trajectories using CaRL's Sokoban
simulator. It sends mock action distributions through the same parser and
rollout as an API controller. The SVG labels the fixture as mock. Result
inputs are frozen in `manifest.json` and `identity.json`, and original trials
are stored under `trials/`. `points.csv`, `report.json` and `reachability.svg`
show the observed denominator and pointwise 95% Wilson intervals. Rerunning
the same command resumes existing trials. Select a new output directory for
any change in the frozen data or configuration. Offline compute and adaptation
times are separate nullable fields.

## Prepare the full campaign

[Campaign configuration](reachability_campaign.json) requests 100 valid
independent layouts per point with up to 20 additional layouts for technical
errors. OpenAI zero-shot, OpenAI four-shot and both local checkpoints run at
`k = 1, 2, 4, 8, 16, 32`. The configured Ministral arm runs at `k = 1, 2, 4, 8, 16`.
Each model takes the same seeded order of eligible layouts at a shared offset.
A provider error uses a reserve, and budget exhaustion leaves a partial point.
Every completed rollout, including a miss, counts toward the valid denominator.

Supply two JSON trajectory files at the manifest's relative paths. Each file
has `schema_version` set to `1`, `split` set to `train` or `test`, and a
`trajectories` list. Each element has a unique string `id`, a `board` array of
12 Sokoban ASCII rows of width 12, and an `actions` list of integer directions
in `[UP, DOWN, LEFT, RIGHT]` order. Actions must be a legal witness from the
initial board to a solved board. The train file must contain at least four
independent four-box layouts for four-shot. The test file needs at least 120
unique ten-box layouts with an eligible non-identical future state for every
configured offset. The validator checks legal transitions, layout identity
under rotation and reflection, and the declared family. Public trial pairs
exclude the reference actions. The test dataset is supplied separately from
model training and in-context examples.

Local BERT needs a `checkpoint-N` directory with `config.json` and
`model.safetensors` from the four-action CaRL CLLP trainer. Local Qwen needs the
base revision, tokenizer, named causal PEFT adapter and hashes from its
[checkpoint config](sokoban_qwen3_lora_eval.example.json). Install `lora` for
Qwen. Local policies use their deterministic four-action argmax. Each local
condition also needs a training receipt with `training_split_fingerprint`
from the frozen train dataset and `policy_fingerprint` from the exact local
checkpoint as printed by `scripts.evaluate_full_state_policy`. Create the
receipt from the data actually used to train or adapt the checkpoint. For
training inputs, see the [BERT pilot config](sokoban_bert_cllp_pilot.yaml) and
[local LoRA commands](../../../scripts/README.md#local-checkpoints-and-lora).

```bash
campaign=configs/icl/full_state/reachability_campaign.json
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config "$campaign" --prepare-only --output "$work/campaign"
uv run --locked --offline --no-sync python -m scripts.run_full_state_reachability \
  --config "$campaign" --model bert --output "$work/campaign"
```

`--prepare-only` checks data and freezes the resolved configurations without
loading local weights or contacting providers. The train hash for the receipt
is `split_fingerprints.train` in `manifest.json`.

## Estimate and explicitly enable API execution

An OpenAI arm has at most 7,560 controller decisions including reserves.
Ministral has at most 3,720. `--plan-only` displays the request and token
ceilings implied by configuration. `--prices prices.json` takes an object
keyed by arm ID with `input_usd` and `output_usd` per million tokens. The
cost estimate multiplies the token ceilings by these rates assuming no cache
hits. Without a rate the estimate is null. For arithmetic verification,
assigning illustrative rates of $1 and $5 per million tokens to an OpenAI arm
gives $125.7984. Actual billing uses the campaign ledger's provider-specific
price and reservations. Local inference and adaptation have separate timing.

Prepare the ignored `configs/icl/campaign_budget.json` using the
[budget template](../campaign_budget.example.json), price the selected
Responses or Chat Completions route and set `CARL_CAMPAIGN_STAGE` to the
corresponding allocated stage. Set the endpoint credential (`OPENAI_API_KEY`
or `MISTRAL_API_KEY`) and choose exactly one API condition. Use the same
ledger across resumed runs. Review the `--plan-only` output first.

```bash
uv run --locked --no-sync python -m scripts.run_full_state_reachability \
  --config "$campaign" --model openai-four-shot --execute-live --output "$work/campaign"
```
