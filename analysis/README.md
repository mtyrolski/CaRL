# Local experiment artifacts

This directory contains local results, complete request/response caches, cost
ledgers, traces, figures and trained weights. These files are preserved on disk
and excluded from Git. The campaign journal is the only tracked run record here.

| Location | Contents |
| --- | --- |
| `luna_icl/campaign_100/campaign.md` | Decisions, measured results and historical commands |
| `luna_icl/campaign_100/generator_probe_v2/` | Calibrated 26-example generator checkpoint and component measurements |
| `luna_icl/campaign_100/goal_progress_probe/` | Goal-placement demonstrations, explicit/verbal probes and offline report |
| `luna_icl/campaign_100/cost_ledger.sqlite3` | Shared campaign charges and reservations; never delete to reset spending |
| `luna_icl/cache/` | Unlimited local response cache; a cached request does not establish run completion |
| `trained_reference/random100_k841/` | Historical native AdaSubS cohort and replayed outcomes |
| `notebook_executions/before_source_cleanup_20261103/` | Executed notebooks preserved before clearing source outputs |

The launch inputs under `configs/icl/launch100/` are small, versioned snapshots.
They contain training demonstrations and initial test states, not test solutions.
Raw research artifacts remain separate from these inference inputs.

Do not infer that an experiment ran from the existence of its configuration,
plan, cache or output directory. Use its completion record and verified replay.
