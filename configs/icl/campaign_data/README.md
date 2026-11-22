# Experiment configurations

These configurations describe component and calibration settings for
in-context search. They inherit shared defaults from `configs/icl/experiments/`.
Use training and validation data for configuration selection, and keep held-out
layouts separate. Data, model outputs and request caches are local inputs.

One-off campaign manifests and their configuration dependencies are preserved
at commit `313a57c403ce9531aea321155bb85fa228d42d65`. To reproduce one of those
runs, open a separate checkout at that commit and supply its original local
inputs.

```bash
git worktree add --detach ../CaRL-pr7-campaigns 313a57c403ce9531aea321155bb85fa228d42d65
```
