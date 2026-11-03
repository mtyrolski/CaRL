# Experimental verbal relative subgoals

This is a separate prototype, not the frozen explicit-board search. The generator
returns short natural-language goals. The controller sees the current board and
goal and outputs one action. The real CaRL environment executes it. After every
move, including a task-solving move, the model judge sees the resulting board and claims whether
the goal was reached, together with an evidence reason. A positive
claim admits the actual endpoint; a negative claim continues until `L`. The
judge is not secretly checked against an exact target. Its false positives and
false negatives remain possible. Task success is checked separately through the
environment; every executed action, including no-ops and fallback actions, uses
the same action budget. The environment decides task success independently even when the final judge call fails. The priority order, shorter-path reopening, and primitive fallback follow
the explicit search structure.

An explicit-board controller's calibrated percentile labels are **not** valid
labels for this controller/judge. Run separate semantic calibration on training
trajectories first. A hit is the judge's claim, not a verified semantic truth;
`d` is its first-hit time and `u` is the within-context midpoint of the weighted
CDF. The present controller chooses a single deterministic action, and local
cache reuse means repeated trials are not independent model samples. With
`R=1` this is a coarse exploratory calibration; insufficient coverage blocks
search rather than inventing labels. Training demonstrations for the policy and
judge are derived from verified training source/target boards, with no simulator
successor probes hidden from the user.

Examples of commands from the repository root:

```bash
uv run --extra icl python -m scripts.verbal_icl.runner plan
uv run --extra icl python -m scripts.verbal_icl.runner calibrate --provider mock
# Real calls use the shared campaign `reserve` stage and require explicit execution.
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.verbal_icl.runner calibrate --provider openai --execute
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.verbal_icl.runner run --provider openai --execute
```

`configs/icl/verbal/smoke.json` evaluates two full validation boards; the
separate `configs/icl/verbal/pilot50.json` is a prospective 50-board preset,
not a result or an authorization to start a large experiment. Use `--config`
to select it. It requires matching semantic calibration and shares the existing
campaign ledger's `reserve` cap. The frozen explicit-state pilot, its caches,
and its manifests are not modified. No live calls were made by preparing this
prototype.

Small goal-placement validation uses the six verified training snippets from
`goal_probe`, with three distinct multi-step contexts and two candidates each.
The first candidate is the pre-placement board described in words; the second
names the newly occupied goal square. These are newly calibrated goals, not old
exact-state percentile labels. Controller examples show the first and last
actual actions of each training snippet. Judge examples show observed outcomes.

```bash
uv run --extra icl python -m scripts.verbal_icl.probe plan --provider openai
uv run --extra icl python -m scripts.verbal_icl.probe calibrate --provider openai --execute
uv run --extra icl python -m scripts.verbal_icl.probe run --provider openai --execute
```

Default: four public queries, one generated goal each, at most eight actions.
`--cases` accepts only 2–6. Checkpoints freeze training data, demonstrations,
prompts, configuration and public queries. Raw JSONL records complete request
payloads before calls and complete responses after calls, in addition to the
shared durable API cache/ledger. Interrupted cases require inspection. No
private validation witness enters these calls. False positives/negatives cannot
be automatically classified without a separate human semantic audit; reasons,
source and actual boards are retained for that audit.

The tiny preset explicitly sets `generator_examples=6` while retaining
`few_shot=3` as the minimum number of distinct calibration contexts with hits.
All successful targets are kept in context-balanced round-robin order, so both
preparation and placement targets can appear. Missing hits are never assigned
invented labels. The signature binds selection policy and example limit.
Raw journals and checkpoints use atomic replacement with file and directory
fsync. The probe always closes its underlying API client in a `finally` block.

The tiny probe loads the unchanged `relative_u_retry.json` training/configuration
for its frozen reference component, verifies its fingerprint even in `plan`, and
previews V from original full-task trajectory continuation labels. Short verbal
snippets are used only for verbal demonstrations/calibration: their lengths are
never treated as remaining actions to task completion. The verbal configuration
has no inherited exact-state policy lock; its independent signature includes the
full frozen reference contract as well as the verbal inputs.
