# In-context planning

The `carl.icl` package provides model-backed components and simulator-based
search. Components are separated by responsibility:

| Package or module | Responsibility |
| --- | --- |
| `llm_components` | Generator, low-level policy and value interfaces |
| `planning` and `search` | Search records, queues and simulator execution |
| `calibration` | Controller-based duration and scale measurements |
| `checkpoints` | Fingerprinted demonstration and calibration inputs |
| `data_sources` and `datasets` | Board and trajectory preparation |
| `intent_data`, `intent_augmentation` | Structured predicates and symmetry transforms |
| `evaluation` | Statistics, reports and board graphics |
| `api`, `responses`, `campaign_budget` | Model transport, caching and request budgets |

Explicit-state generators propose complete target boards. Their controllers
receive the target board and check exact state matches. Structured intents are
separate predicate requests; changing a stopping condition alone does not change
a controller's input contract.

A primitive-action budget includes attempted actions, failed rollouts and
no-ops. Expansion counts, model calls and wall time are separate measurements.
Keep training, validation and evaluation layouts disjoint, including symmetries.

See [`ICL_SEARCH.md`](../../ICL_SEARCH.md) for commands and configuration.
