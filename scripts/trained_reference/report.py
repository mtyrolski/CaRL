"""Report historical measurements without turning replays into new solver runs."""
from pathlib import Path
from typing import Any

from scripts.trained_reference.common import save, wilson, write_csv


def write_report(output: Path, rows: list[dict[str, Any]], population: dict[str, Any]) -> None:
    solved = sum(row['solved'] for row in rows)
    low, high = wilson(solved, len(rows))
    summary = {'cohort': 'historical_random100_k841', 'boards': len(rows), 'solved': solved,
               'unsolved': len(rows) - solved, 'solve_fraction': solved / len(rows),
               'wilson_95_low': low, 'wilson_95_high': high, 'native_low_level_cap': 500,
               'records_over_native_cap': sum(row['native_overshoot'] > 0 for row in rows),
               'maximum_recorded_low_level_steps': max(row['low_level_nodes_visited'] for row in rows),
               'replayed_solutions': sum(row['solution_replayed'] is True for row in rows),
               'new_solver_runs': 0, 'api_calls': 0,
               'current_pilot_board_overlap': population['current_pilot_board_overlap']}
    save(output / 'summary.json', summary)
    write_csv(output / 'summary.csv', [summary])
    write_csv(output / 'reference_outcomes.csv', rows)
    save(output / 'population_audit.json', population)
    (output / 'report.md').write_text(f'''# Frozen historical AdaSubS reference: random 100 boards

The existing pretrained AdaSubS k=(8,4,1) records solve **{solved}/{len(rows)}**
selected real 12×12×4 boards; {len(rows) - solved} recorded attempts fail.
All {solved} saved solution paths were independently replayed in the local
Sokoban simulator. There were **zero new solver runs and zero API calls**.
Replay confirms the recorded witness; it is not a new inference result.

Selection uses NumPy PCG64 seed 20261102, without replacement, from the first
1,000 states of the original progress-board file. The 100 indices and initial
states are frozen in `boards.json` before loading solver outcomes. Selection
uses neither success labels nor solution lengths. All 1,000 historical
attempts were joined to that ordered source population by full-state SHA-256;
there are {population['solved']}/1,000 solved records in the complete archive.
Despite the `solved_problems` filename, failed attempts are included by the
original writer and remain in this sample's denominator.

| Measure | Recorded result |
|---|---:|
| Solved / sampled boards | {solved}/{len(rows)} |
| Recorded failures | {len(rows) - solved} |
| Descriptive Wilson 95% interval | {100 * low:.2f}%–{100 * high:.2f}% |
| Native low-level counter cap | 500 |
| Sample attempts exceeding native cap | {summary['records_over_native_cap']} |
| Maximum observed low-level steps | {summary['maximum_recorded_low_level_steps']} |
| Current Luna pilot board overlap | {population['current_pilot_board_overlap']} |

The native classes are `carl.solver.subgoal_search.Solver`,
`carl.planners.adasubs.AdasubsPlanner`, `AdaptiveSubgoalGenerator`,
`BasicValidator`, `TransformerConditionalLowLevelPolicy` and `TransformerValue`.
The planner prioritizes (-k, -value); deterministic generation uses beam width
4 and one returned proposal. The greedy validator receives explicit expansion
k, which overrides its default cap. The native counter counts executed
low-level validator steps, including rejected proposals. It is checked before
an expansion and can overshoot 500. This is not a strict action-budget B=500
experiment and is not the newer ICL search wrapper. Generator and value calls
are exported separately; no missing runtime or model-token counts are inferred.

Historical exact-tag Neptune configurations agree on k=(8,4,1), beam=4,
returned proposals=1, native cap=500 and checkpoint paths. They are archived in
`config_evidence.json`, with available raw-log hashes. The batch files lack
per-attempt run UUIDs, configuration hashes and training/evaluation seeds.
`freeze.json` records SHA-256 hashes of source batches and currently available
checkpoint weights/configs at those paths; those hashes were measured now and
cannot prove which bytes were loaded by the historical process. Training
membership and overlap with these boards remain unknown. The interval is a
descriptive board-level summary, not a generalization guarantee.

`boards.json` contains public initial states only. `reference_outcomes.csv`
contains all 100 outcomes, source batch/index, low-level work, generator/value
counts, native overshoot and replay status. `records/` additionally stores the
saved action and subgoal paths for inspection. Reference solution paths must
not be placed in Luna prompts. `population_audit.json` records the complete
archive join and current-pilot overlap. Without shared board IDs, no paired
comparison with the old Luna pilot is available; future comparisons must use
this frozen cohort and declare their own budgets and component settings.

Rebuild offline:

```bash
uv run --extra icl python -m scripts.trained_reference.freeze
```

A changed frozen selection or input hash is rejected; use a separate output
location for a new cohort. Source records are read-only and remain untouched.
''', encoding='utf-8')
