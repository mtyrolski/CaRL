# Reachability runner

This runner evaluates a goal-conditioned policy on saved Sokoban trajectory segments. `k` is the number of witness actions between the source and target, not a proven shortest distance. A trial stops on an exact target state (including the player), task completion, or the action limit `L`. A no-op uses one action. Task completion before the target is not a hit.

The dataset and its private witness bundle live under `data/icl/` and are not included in Git. Initial boards alone are insufficient to reconstruct trajectory pairs. Prepare or restore the matching dataset before running `plan` or `preflight`.

```bash
python -m scripts.reachability_runner plan --config path/to/study.json
python -m scripts.reachability_runner preflight --config path/to/study.json
python -m scripts.reachability_runner prepare --config path/to/study.json
python -m scripts.reachability_runner run --config path/to/study.json
python -m scripts.reachability_runner resume --config path/to/study.json
python -m scripts.reachability_runner verify --config path/to/study.json
python -m scripts.reachability_runner export --config path/to/study.json
```

Use a new output directory for each study. Mock runs use simulated responses; replay uses an explicitly selected saved cache and cannot fetch missing responses. Neither constitutes a new live measurement. Resume checks the frozen contract and retains completed outcomes and attempt history.

Live execution requires an explicit model profile, separately authorised limits and a campaign ledger. A request with unknown billing remains reserved and blocks automatic retry until reconciled. Do not commit request or response caches, ledgers, credentials, or private witnesses. Historical campaign-specific recovery scripts and incident records are separate from this runner.
