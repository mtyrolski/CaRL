# Archived experiment configurations

Use `configs/solve/`, `configs/offline_training/` and `configs/icl/` for current runs. The archive keeps representative configurations for older methods, architectures and input/output contracts. Their original code, data or checkpoints may be needed to reproduce those runs. The historical Sokoban BFS recipe contains two `carl_grid` entries and requires choosing a grid before Hydra can compose it.

Keep a single maintained configuration for each domain, method and model contract. Put seeds, dataset shards, checkpoint selections, search limits and other run parameters in Hydra overrides or a campaign manifest. Use `carl_grid` for a parameter sweep and keep the resolved configuration, Git revision and exact command with its results under `analysis/`. Add a shared configuration when a new method, architecture or interface needs one.

Preview the configuration before scheduling a run. This example composes a single NPuzzle AdaSubS run with a two-problem limit.

```bash
uv run --locked python -m carl.run \
  --config-dir configs/solve/npuzzle --config-name npuzzle_ada_solve \
  --cfg job 'carl_grid=[{}]' algorithm.problems_to_solve=2
```

The entire archive from before consolidation is available at revision
`d37b422643cc9b79af49d1fe691478457545681f`. Extract it into a separate directory to recover the original grids, comments and checkpoint selections.

```bash
mkdir -p analysis/config-archive/d37b422
git archive d37b422643cc9b79af49d1fe691478457545681f configs/archive \
  | tar -xf - -C analysis/config-archive/d37b422
```

For an existing result, use the revision recorded with that run when available. For a new campaign, store its exact overrides and resolved input alongside the result so the shared configuration can evolve independently.
