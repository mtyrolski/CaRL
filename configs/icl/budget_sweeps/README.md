# Action-budget sweeps

A sweep specifies a matrix of conditions, independent action budgets and an
output directory. Inspect a configuration without running inference:

```bash
uv run --extra icl python -m scripts.run_budget_sweep --config configs/icl/budget_sweeps/mock.json --plan
```

Use `--mock` for an offline interface check, `--replay` for cached responses,
and `--execute` for live model calls. `--resume` verifies the saved run contract
before reusing completed trials. Input data and any required calibration must be
prepared separately. Use `scripts.evaluate_icl` to analyse saved runs offline.
