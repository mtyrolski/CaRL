# Intent pilot status

## Prompt 01 — Campaign bootstrap, 2026-11-22

Complete. The checkout was `intents` at `5ddd4cb7bcde591fdcf7a0f32b0aa41b460e8b0c` with a clean tree at intake. Both earlier reference commits are ancestors. The local November 14 proposal supplied the Method and Evaluation.

| Proposal source | SHA-256 |
| --- | --- |
| `03_method.tex` | `1b9442aebc194bf3264b726331ce6c24ac2c79c476ef84e40338676889864b02` |
| `04_evaluation.tex` | `bfce0ce8bc752f65473a346ed4b7af4aa602b6df32569cf4f2f9873b96d174e9` |

### Frozen inputs and outputs

| Partition | Sokoban complete source episodes | Craftax scheduled native episodes | Craftax canonical chains |
| --- | ---: | ---: | ---: |
| Train support | 8 | 12 | 6 |
| Train measurement | 9 | 6 | 3 |
| Validation | 4 | 4 | 2 |
| Sealed test | 8 | 4 | 2 |

Sokoban uses the existing campaign demonstrations and their witnesses, the original gamma1 witnesses, the generated 12×12 ten-box validation witnesses and the gamma2 whole-episode pool. The manifest records source file paths and hashes. The Craftax private roster pins native scenarios, ordered milestones and two independently seeded worlds per chain. No segments or new labels have been produced. The first native Craftax collection will check snapshots and task feasibility on the frozen worlds.

- Split manifest `analysis/intent-pilot/splits.json`, SHA-256 `5c6f2810968c5d622550f519c11d6ad1d3ac6090cae984fad6e60a79d27ffd7f`
- Train and validation whole-episode files in `analysis/intent-pilot/splits/`
- Evaluator-only test files in `analysis/intent-pilot/sealed/`. The sealed directory and private inputs have mode `0700`.
- Budget ledger `analysis/intent-pilot/campaign.sqlite3` and read-only snapshot exports `stage01-budget.json` and `stage01-budget.csv` under the same root
- Verified resume summary `analysis/intent-pilot/stage01-resume.json`, audit `verification.json`, environment inventory `environment.json`, and logs under `logs/`

### Local environment

The WSL2 environment exposes an Intel i9-12900HK, 20 logical CPUs, 10 guest-visible cores and 58,969,616,384 bytes of RAM. GPU access could not be verified in the sandbox. The recorded runtime selects CPU. The bootstrap uses uv 0.12.21, CPython 3.11.4, Torch 2.10.0, Transformers 4.57.6, PEFT 0.21.3, NumPy 1.26.4, Pydantic 2.13.5 and Safetensors 0.7.0. Craftax 1.1.0 uses JAX and JAXLIB 0.4.23. The complete version inventory and tokenizer file hashes are in `environment.json`.

| Controller | Local state |
| --- | --- |
| Qwen/Qwen3-0.6B | Cached at `analysis/intent_generator_20261114T032232Z/models/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. The 1,503,300,328-byte weight file has SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. |
| Qwen/Qwen3-1.7B | Snapshot and revision are unresolved. The checked Hugging Face cache and project model directories contained no snapshot. A verified local snapshot is required before this arm can run. |

The lockfile SHA-256 is `725622210195c95d959efbc418cba3b4241b6ec33b56a486991a058dc5d5bac2`. The separate `requirements/craftax-1.1.0.txt` pins hash to `7f9c0ba5f5b82e5296d09567f254727f4cb4c341fddea318ef59d8c4f8cc55d7`. Source `analysis/intent-pilot/env.sh` from the repo root to reuse the existing locked environment and Craftax package overlay.

### Verified entry points and results

`scripts.intent_pilot` initialized the ledger, froze and verified both domains, and recorded the bootstrap invocations. Rebuilding produced the same split hash. The audit verified group separation, omission of sealed paths from the resume summary, admission under competing reservations, access to the protected Prompt 09 budget, and charging for failed and timed-out commands. Ruff on the new modules and `git diff --check` passed.

The initial freeze stopped on the quick-start query's raw board-list format. The corrected fixture reader completed the retry, and the ledger charges both attempts. CPU inference, native controller rollouts and model downloads have not started. Human and Codex annotation time are separate preparation rows. Per-run Codex usage is unknown.

| Bootstrap job | Result | Worker-seconds |
| --- | --- | ---: |
| Freeze whole episodes | Failed on fixture format | 1.453 |
| Freeze whole episodes retry | Complete | 1.777 |
| Rebuild, budget and cache audit | Complete | 3.720 |

**Remaining budget** is USD 0 metered and 7,193.050 local worker-seconds, of which 5,393.050 can be committed by Prompts 01–08. Prompt 09 retains 1,800 seconds. Inference and outstanding reservations are both zero. These figures describe the ledger at the end of this stage. Query `status` for the live balance.

### Exact resume command

From the CaRL checkout, after reading [CONTRACT.md](CONTRACT.md) and this journal, run the tested verification entry point.

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot verify \
  > analysis/intent-pilot/stage02-input-check.json
```

Inputs are the pinned manifest, its split files and the bound SQLite ledger. The command writes hashes, counts, unsealed input paths and the current budget to `stage02-input-check.json`. The sealed files are checked by hash.

### Next prompt — 02

Read both documents and run the resume command. Prepare replay-checked annotation inputs for both domains from the frozen train-support episodes. For Sokoban, pass the `episodes[].demonstration` records as `trajectories` to the existing replay and annotation contract in `carl.icl.intent_annotations.prepare_annotations`. The default CLI sketch provider and its mock option are integration checks. Use Codex's own checked labels for this campaign.

For Craftax, construct the pinned simulator with `CraftaxAdapter`, then run native `reset`, `stage`, `step` and `restore` on the unsealed worlds. Record actual world identities, replayability and termination before selecting fragments. The existing `scripts.run_predicate_icl craftax` CLI exposes `--mode mock` for the published fixture. The local real-controller surface for this campaign needs to be connected to the native Craftax executor before inference. Keep observations, actions and outcomes separate from model-visible requests.

Budget all replay, integration and verification jobs through the ledger. Keep train-measurement, validation and sealed test separate. Record checked Codex annotation provenance for both domains, then append the exact inputs, outputs, commands, results, remaining budget and the next prompt to this journal.

## Prompt 02 — Integration smokes and checkpoint replay, 2026-11-22

Complete on the intake checkout `8be17acfcd5f9253a17cadf0bf4d1f275f06ae8e`. Both smokes and the replay audits are **integration-only**. They used the public Sokoban quick-start fixtures and the pinned native Craftax task `wood-to-stone-pickaxe`. The campaign split hash reproduced and the sealed cohorts stayed outside these runs.

### Commands and artefacts

From the CaRL root, source `analysis/intent-pilot/env.sh` and set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The native commands also set `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`. The existing Craftax 1.1.0 and JAX 0.4.23 overlay follows the package setup in `scripts/README.md`. All jobs used `uv run --locked --offline --no-sync` inside `scripts.intent_pilot run --prompt 2 --phase preparation`. The ledger contains the exact argv, deadlines and costs.

| Entry point and arguments | Output under `analysis/intent-pilot/smoke/` |
| --- | --- |
| `python -m scripts.run_icl_smoke --config configs/icl/quickstart/sokoban_sweep.json --output analysis/intent-pilot/smoke/sokoban` | `sokoban/sweep.json`, per-trial traces, `sokoban.log` |
| `python -m scripts.run_predicate_icl craftax --mode mock --planning-mode all --task wood-to-stone-pickaxe --output analysis/intent-pilot/smoke/craftax.json` | `craftax.json`, `craftax.log` |
| `python -m scripts.intent_pilot replay --domain sokoban --output analysis/intent-pilot/smoke/sokoban-replay.json` | `sokoban-replay.json`, `sokoban-replay.log` |
| `python -m scripts.intent_pilot replay --domain craftax --output analysis/intent-pilot/smoke/craftax-replay.json` | `craftax-replay.json`, `craftax-replay.log` |
| `python analysis/intent-pilot/smoke/reconcile.py` | `integration-only.json`, `action-work.csv`, `reconcile.log` |

The Craftax entry accepts `mock` and its `mock_proposals` returned identical proposals at requested scales 0.2 and 0.8. The native output carries the integration-only label. The reconciled JSON labels both smokes and binds their raw files by SHA-256.

| Fixture smoke | Solved | Policy / generator / value calls | Live / speculative ticks | Warm-up ticks |
| --- | ---: | ---: | ---: | ---: |
| Sokoban | 22 / 24 | 44 / 16 / 8 | 20 / 24 | 0 |
| Native Craftax | 4 / 4 | 25 / 9 / 7 | 15 / 10 | 1 |

The two Sokoban one-step runs at the smaller budget stopped on `action_budget`. In both smokes all primitive requests in the matched records executed, and the matched records contain zero automatic ticks. The Craftax warm-up is charged separately from its matched records. Direct, committed one-step and Plan-and-Act execution are live for fixture accounting. Rollout probes and full-search expansions are speculative, including the selected full-search path. All role calls were mocked and physical requests were zero.

### Replay and native tick evidence

- Both domains cloned a checkpoint with one prior action in controller memory. The restored branches matched the reference observations, decisions, transitions and endpoint memory. Sokoban reached its first checked hit after one executed primitive from the checkpoint. Craftax reached it after four, with identical reward, terminal flags and full snapshot fingerprints.
- The independent Craftax stream probe changed the future native key while holding the full initial physical state, visible input, controller seed and `DO` decision fixed. Seven bounded branches yielded both zero and one sapling. The evaluator logs record fingerprints and branch identifiers for the RNG check.
- A native sleep tick overrode the requested `DO`. It recorded one automatic tick, zero executed primitives and verified wood preservation. A second probe ran two native automatic ticks and withheld their intermediate observation from the checker. It returned `unknown` with no first-hit duration. `action-work.csv` and the replay JSON retain the per-case work class, decisions, role counts, tick counts and stop reasons.
- The two actual replays passed. The targeted locked regression command ran `tests/experiments/test_luna_intent_rollout.py`, `test_predicate_intent_search.py` and `test_craftax.py` with `-k 'not native_craftax'`. It passed 33 cases and deselected three native cases. The public smoke and checkpoint replay supplied the native checks. Ruff and `git diff --check` passed.

### Reconciled budget

| Job | Worker-seconds |
| --- | ---: |
| Sokoban predicate smoke | 18.693 |
| Native Craftax smoke, including warm-up | 117.271 |
| Sokoban checkpoint replay | 10.556 |
| Craftax checkpoint and stochastic replay | 119.166 |
| Targeted regressions | 11.490 |
| Integration reconciliation | 0.812 |
| Prompt 02 total | 277.988 |

All six jobs settled. The stage charged **277.988 preparation worker-seconds** and zero inference worker-seconds. Cumulative preparation is 284.938 seconds. The balance is **6,915.062 seconds**, with **5,115.062 available before Prompt 09** and **1,800 protected for Prompt 09**. Outstanding reservations and metered USD are both zero. The full snapshot is `analysis/intent-pilot/stage02-budget.json`.

### Exact resume command

From the CaRL root, read [CONTRACT.md](CONTRACT.md) and this journal, then verify the existing split package against the ledger.

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot verify \
  > analysis/intent-pilot/stage03-input-check.json
```

This reads the frozen split package and bound ledger. It writes the current balance, counts, hashes and unsealed input paths. The completed verification is saved as `analysis/intent-pilot/stage02-resume.json`.

### Next prompt — 03

Read the contract and status, then run the resume command. Prepare replay-checked annotation inputs for both domains from the frozen train-support episodes. Validate the unsealed Sokoban witnesses and collect native Craftax transitions on their pinned worlds with restorable snapshots. Use Codex's own typed annotations and keep annotation evidence, automatic ticks and unknown outcomes in the audit. Reserve calibration and validation data for their declared roles and keep fixture traces and sealed chains out of training and model inputs. Record preparation costs, artefacts, the result and the next exact resume command in this journal.

## Prompt 03 — First checked demonstrations, 2026-11-22

Complete on the intake checkout `61b990963f2f6a0dd111ec78dbc6480ae89cd9ed`. Each domain supplies **24 accepted train-support fragments** from eight independent groups, with three endpoints per chosen start. A separate set of four Sokoban layout groups and two Craftax chains supplies validation.

| Domain | Independent support groups | Raw packets | Accepted train | Accepted validation | False | Unknown | Malformed | Duplicate | Abstained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sokoban | 8 | 36 | 24 | 12 | 0 | 0 | 0 | 0 | 0 |
| Craftax | 8 | 30 | 24 | 5 | 0 | 0 | 0 | 1 | 0 |

### Native inputs and annotation

- Sokoban replays 12 complete frozen local witnesses, covering 1,577 source actions. The eight independent support units are distinct source episodes and canonical layouts within the original real four-box source family. The archived witnesses supplied the full target yield. Twenty-two accepted training rows fill at least one new goal and verify that an already occupied goal stays occupied through the first hit.
- Craftax replayed 39 scripted native actions over ten seeded worlds, then reproduced each episode from its reset and staged snapshot. The support collection plan freezes the eight worlds and crafting chains before collection. It binds to the base split and excludes withheld worlds, chains, and withheld chains embedded in a support sequence. Two previously frozen validation worlds supply the remaining packets. The first collection used the wrong facing direction on a validation world. The corrected demonstrator reads the stone direction from the frozen scenario. The failed run, its original plan, and the successful retry remain in the ledger and artefacts.
- Codex read the endpoint packets in four Sokoban batches of nine and five Craftax batches of six, then authored one raw JSONL response per packet in this session. The prompt is saved verbatim with SHA-256 `8ba9e04b8b69e174d448799f6c2568389657fcef2c8177ab2197511be9ff207e`. The manifests record Codex, GPT-6 family, and an unknown exact build. The observed annotation windows were 180 seconds for Sokoban and 144 for Craftax, in separate preparation rows. Per-session tokens are unknown. Sokoban witnesses, the Craftax scripts and Codex labels have distinct provenance. All 66 labels came from Codex.

Each `D_A.jsonl` has partial predicate intents with model-visible start observations. The evaluator audit retains original endpoints, the verified prefixes, snapshot and memory references, and reasons for rejected rows. Sokoban uses the actual per-packet responses through the `prepare_annotations` callback. Craftax uses `CraftaxIntent` and the replayed native inventory at every tick. There are no measured scale labels. The duplicate came from two different facing observations that support the same sword-tier intent in the same validation context.

### Outputs and verification

Each domain directory under `analysis/intent-pilot/annotations/` contains `D_A.jsonl`, `annotations.raw.jsonl`, `annotation_audit.jsonl`, `split_manifest.json`, `manifest.json`, and the hashed blinded packets. The protected `private/` directories contain complete native traces, script or source provenance, and the packet-to-trajectory index. Representative accepted rows are in `annotations/labelled-yield.json`.

| Domain | Split manifest SHA-256 | Checked dataset SHA-256 |
| --- | --- | --- |
| Sokoban | `f6ced20885d191b02d152e22d02b592bb9f85790c0df9b170d6d7651272e0694` | `a6e815621d10b5c7661ff53920929c8e1a0253610a28db997a1683fa6228b940` |
| Craftax | `fd62e3f96be616ad198bddff87ad6c9e6cfe20bfa6fda7725ebb52f595116331` | `9c45ef41a9df8c011e88735d6baa807358438994d6cc114e5b7d71f760213ae0` |

The executed collection entry point was `python -m scripts.intent_pilot annotations collect --domain DOMAIN` inside the locked environment and the Prompt 03 preparation ledger. The executed rebuild entry point was `python -m scripts.intent_pilot annotations build --domain DOMAIN`. Both domains passed `annotations verify`, which independently rebuilds the checked bytes from the frozen evidence and raw responses and compares them to the saved outputs.

The targeted locked regression run passed 33 tests. It covered the legacy annotation and policy-measurement consumers, selected-prefix preservation with an independently checked original endpoint, missing Craftax tick evidence, and rejection of extra scale fields. Ruff and `git diff --check` passed.

### Reconciled budget

Prompt 03 charged **308.863 preparation worker-seconds**, including 147.547 for the failed Craftax collection and 94.964 for its retry. Cumulative preparation is 593.801 seconds. **6,606.199 worker-seconds remain**, with **4,806.199 available before Prompt 09** and **1,800 protected for Prompt 09**. Outstanding reservations, inference worker-seconds, and metered USD are zero. Exact invocations and costs are in `analysis/intent-pilot/stage03-budget.json`. Stage logs are under `analysis/intent-pilot/logs/stage03-*.log`.

### Exact resume command

From the CaRL checkout, read the contract and journal, then verify the base split and both checked datasets with the same ledger.

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot verify \
  > analysis/intent-pilot/stage04-input-check.json
for domain in sokoban craftax; do
  uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
    --prompt 4 --domain "$domain" --phase preparation --activity verify-checked-demonstrations --seconds 90 -- \
    uv run --locked --offline --no-sync python -m scripts.intent_pilot annotations verify --domain "$domain" \
    > "analysis/intent-pilot/stage04-$domain-dataset-check.json" || break
done
```

The inputs are the locked split and ledger, the per-domain packet manifests, actual raw responses and private replay evidence. The checks write the current budget and verified dataset hashes without copying private trace contents. `stage03-resume.json` and the per-domain verify logs record the completed checks for this stage.

### Next prompt — 04

Read both documents and verify the checked datasets. Use the `train-support` rows for demonstration support and the frozen `train-measurement` cohort for controller trials. Pin each real local controller's model, tokenizer, action interface, memory and decoding before measuring. Record the currently unresolved Qwen3-1.7B snapshot as a preparation dependency. Keep validation and sealed groups separate, then append the actual inputs, commands, output paths, results and reconciled budget to this journal.

## Prompt 04 — Real local direct control, 2026-11-22

The intake checkout was `intents` at `41541dbc138b7e0d7c78dcef18d3493764df81c4`. Both checked demonstration datasets and the ledger-bound split passed verification. The direct roster was frozen before the first model saw a task. Its SHA-256 is `f4192d9041cb0920031634841525593703f24ef018e3de8a3031cee73e866e00`.

Each controller received two Sokoban walk-to-push-position starts and two fill-and-keep-goal starts from four distinct validation layouts. In Craftax it received two craft-wood-sword and two gather-wood starts from four distinct validation worlds. The staged Craftax starts share the local terrain and inventory needed to compare those skills. All sixteen scores use validation. The public `collect-wood` CLI fixture is a separate integration record.

| Requested checkpoint | Actual checkpoint | Revision |
| --- | --- | --- |
| Qwen3-0.6B | Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3-1.7B | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

Qwen3-0.6B ran first. Offline snapshot recovery and workspace inspection found no complete Qwen3-1.7B checkpoint. The named substitute was already cached. Both used the pinned Python 3.11.4 environment, Torch 2.10.0, Transformers 4.57.6, CPU float32 and six Torch math threads on the Intel Core i9-12900HK. The native adapter used Craftax 1.1.0 and JAX 0.4.23. The loader checked the revisions, weight and tokenizer hashes and compared each rendered chat's tokens with the actual input tensor. Qwen3's template changes when thinking is toggled. Qwen2.5's template renders the same with the flag passed.

Every trial used greedy decoding, at most 96 generated tokens per decision, a 1,024-token context cap, three decisions, six executed primitives and eight native ticks. The 60-second trial guard is checked between model calls and actions. The ledger applies a hard process deadline. The controller can supply an intent and up to four legal actions. Each interruption discards the remaining suffix. The next decision receives the current public observation and the last six executed-action records. Every executed no-op consumes a primitive. Each completed trial is written atomically. The process shares one loaded policy model across domains, with a fresh action history per start.

The public Craftax local CLI loaded Qwen3-0.6B and dispatched one model decision on the `collect-wood` fixture. It returned `RIGHT`, charged one executed no-op, recorded 527 input and 15 output tokens, and exited 2 at the decision limit. The existing Sokoban local path and the Craftax mock dispatcher retain their original entry points. The new regression exercises interruption, history, suffix disposal and first-hit action charging. The locked base `make test` passed 629 tests with four skips. Native Craftax was exercised separately by the measured trials and the public CLI fixture.

### Executed entry points

The preparation wrapper verified both `annotations verify --domain DOMAIN` datasets, froze `direct freeze`, and recorded the model load probe and cached recovery. The following uses the saved configuration and artefact names. Existing completed result files are protected from overwrite.

```bash
source analysis/intent-pilot/env.sh
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
for controller in qwen3-0.6b qwen2.5-1.5b-instruct; do
  uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
    --prompt 4 --phase inference --activity "${controller}-direct" --seconds 600 -- \
    uv run --locked --offline --no-sync python -m scripts.intent_pilot direct run \
    --controller "analysis/intent-pilot/direct/${controller}.json" \
    --output "analysis/intent-pilot/direct/${controller}-results.json"
done
```

The initial Qwen2.5 process reached its hard deadline after four Sokoban starts and one Craftax start. The ledger kept the full timed-out charge. The interrupted report was copied to `direct/qwen2.5-1.5b-instruct-interrupted.json`. The continuation checks the frozen roster and actual model binding and preserves the completed prefix. A separately recorded neutral model warm-up restored the warm weights before the remaining native starts.

```bash
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase inference --activity qwen2.5-1.5b-instruct-direct-resume --seconds 660 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot direct run --resume \
  --controller analysis/intent-pilot/direct/qwen2.5-1.5b-instruct.json \
  --output analysis/intent-pilot/direct/qwen2.5-1.5b-instruct-results.json
```

```bash
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --domain craftax --phase preparation --activity craftax-public-local-cli --seconds 440 -- \
  nice -n 8 uv run --locked --offline --no-sync python -m scripts.run_predicate_icl \
  craftax --mode local --planning-mode direct --task collect-wood \
  --local-controller analysis/intent-pilot/direct/qwen3-0.6b.json --max-requests 1 --action-budget 4 \
  --output analysis/intent-pilot/direct/craftax-cli-local.json
```

The base regression used `PYTHONPATH="$PWD"`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `PYTEST_ADDOPTS="-q -p no:cacheprovider"`. The ledger ran `nice -n 8 make test 'UV_RUN=uv run --locked --offline --no-sync'` with a 780-second deadline. An initial regression run found that the newly added test fixture was rectangular. The corrected square fixture passed both its focused test and the full base test.

### Results and verification

| Actual controller | Game | Successes | Errors | Loops | Policy calls | Executed primitives / no-ops | Input / output tokens | Model seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | Sokoban | 0/4 | 0 | 0 | 12 | 12 / 5 | 4,264 / 159 | 94.905 |
| Qwen3-0.6B | Craftax | 2/4 | 0 | 0 | 12 | 12 / 10 | 6,918 / 272 | 127.479 |
| Qwen2.5-1.5B-Instruct | Sokoban | 0/4 | 7 | 0 | 11 | 4 / 1 | 3,814 / 114 | 265.515 |
| Qwen2.5-1.5B-Instruct | Craftax | 0/4 | 8 | 0 | 12 | 4 / 0 | 6,770 / 124 | 231.881 |

The first cold decisions took 24.202 seconds for Qwen3 and 59.935 seconds for Qwen2.5, including 4.322 and 22.062 seconds of model stack loading. The warm median was 5.320 seconds on Sokoban and 10.935 on Craftax for Qwen3. The corresponding Qwen2.5 medians were 22.202 and 23.999 seconds. Qwen2.5's separate resume warm-up took 53.778 seconds and 33 input and 2 output tokens. Native setup took 240.448 seconds for Qwen3 and 408.297 for Qwen2.5 across its two processes. The score table counts completed trial calls. The timed-out interval after the last atomic result and the warm-ups are charged at the process level. All scored native ticks came from executed requests.

| Prerequisite | Qwen3-0.6B | Qwen2.5-1.5B-Instruct | One-action evaluator check for misses |
| --- | ---: | ---: | --- |
| Sokoban walk-to-push-position | 0/2 | 0/2 | `UP` on `sokoban-1` |
| Sokoban fill-and-keep-goal | 0/2 | 0/2 | `UP` on `sokoban-3` |
| Craftax craft-wood-sword | 2/2 | 0/2 | `MAKE_WOOD_SWORD` on `craftax-1` |
| Craftax gather-wood | 0/2 | 0/2 | `DO` on `craftax-3` |

Qwen3 chose `RIGHT` at every Sokoban decision and repeated `LEFT` against the tree in Craftax. It crafted both swords after two charged no-ops on each start. Qwen2.5 opened with `RIGHT` in seven starts and `DOWN` in the fourth Sokoban start. All 15 follow-up decisions contained empty action batches and failed validation. The evaluator verified the frozen native first hit and preservation conditions. Every model/domain cell missed at least one prerequisite, so this stage stopped at direct validation.

The audit verified all 16 saved trials, including their exact input payloads, common limits, match to the frozen roster, executed action histories, discarded suffixes and first-hit accounting. The per-cell controller fingerprint binds the model files, chat template, protocol sources and limits. Targets for Ruff, Pyright and Mypy passed. The protocol was committed at `263618c`.

```bash
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase preparation --activity audit-direct-validation --seconds 75 -- \
  uv run --locked --offline --no-sync python analysis/intent-pilot/direct/report.py
```

The raw controller configs, results, transcripts, initial and continuation source snapshots, diagnostic audit, human report and CSV are in `analysis/intent-pilot/direct/`. The audit and ledger preserve the timeout, the completed prefix and the continuation. The public fixture record is `analysis/intent-pilot/direct/craftax-cli-local.json`. The regression log is `analysis/intent-pilot/direct/make-test.log`.

### Reconciled budget

Prompt 04 charged **847.382 preparation worker-seconds** and **1,495.033 inference worker-seconds**. Its inference jobs include the itemized native and model setup inside their process envelopes. The first Qwen2.5 job charged 600.504 seconds and the continuation charged 420.476. The one-decision public integration fixture charged 372.377 seconds under preparation. The ledger recorded zero download time, unknown Codex usage, zero metered USD and zero outstanding reservations.

**4,263.784 worker-seconds remain**, including **2,463.784 available before Prompt 09** and **1,800 protected for Prompt 09**. The exact jobs, exits and cumulative balance are in `analysis/intent-pilot/stage04-budget.json`.

### Exact resume command

From the CaRL checkout, read both pilot documents and run the verified input check.

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot verify \
  > analysis/intent-pilot/stage05-input-check.json
```

This reproduced the ledger-bound split SHA-256 `5c6f2810968c5d622550f519c11d6ad1d3ac6090cae984fad6e60a79d27ffd7f`. Use the two pinned controller JSONs, the direct audit and verified `D_A` manifests to plan the next bounded run.

### Next prompt — 05

Read both pilot documents and run the resume command. Start with the checked support data and the frozen train-measurement cohorts. Use the same action and memory interface, require a verified prerequisite hit before allocating larger rollouts to a controller and task, and record unsuccessful or incomplete measurements explicitly. Keep sealed compositions separate. Use the actual controller revisions and reconcile any preparation, calibration or retry time against the remaining budget before writing the next handoff.

## Prompt 04 continuation — Assigned intents and checked hindsight, 2026-11-22

Complete on intake `7fb7009`. The assigned-intent runner and verifier are in `68a65ae`. The collection froze four train-support contexts before the model calls. The private plan SHA-256 is `7bc3c098eaca601f103dd8063a6d96f4e8241138af09b4c04def0f3ceca2a2fd`. Its inputs are the previously checked native training sources, the original split and checked `D_A` hashes, and the two pinned controller JSONs from the direct run.

The runner keeps the assigned `h` in each query across interrupted batches. It checks preservation on the start and each native transition, and records the action, first hit, task and episode state, state and memory identities, and remaining shared budget. An admitted batch can execute up to four actions. The normal cap is 12 executed primitives per intent.

| Controller | Sokoban executed primitives | Craftax executed primitives | Status in both |
| --- | ---: | ---: | --- |
| Qwen3-0.6B | 4 | 6 | `budget_cutoff` after three policy calls |
| Qwen2.5-1.5B-Instruct | 2 | 1 | `budget_cutoff` after three policy calls |

The four real model attempts reproduced under native replay. Their effects remained unmet and none moved a box or increased an inventory count within the selected progress rule. The model binding and exact payloads are saved with each attempt. Qwen2.5 is the pinned local substitute already recorded in the direct stage.

Two separately declared scripted training attempts on frozen `train-support` groups reached `intent_timeout` after exactly 12 primitives. In both natives the replan began from the actual timeout endpoint and memory with 20 actions left. The new requests reached their effects in four primitives in Sokoban and two in Craftax. The integration-only speculative traces instead retained their original frontier checkpoints and charged all 12 executed primitives.

### Checked hindsight and pool

Codex read two blinded packets using the original Pilot annotation instruction with SHA-256 `8ba9e04b8b69e174d448799f6c2568389657fcef2c8177ab2197511be9ff207e`. The exact two raw replies were written in this session. The separate annotation ledger row records the observed 31.872-second window. Token usage is unknown.

| Native source attempt | Requested outcome | Verified hindsight outcome | Observed first hit under the original request |
| --- | --- | --- | ---: |
| `scripted-sokoban` | Fill both specified goals while preserving an existing box | Fill the goal at `(2,2)` while keeping the box at `(2,6)` | 1 |
| `scripted-craftax` | Reach at least three stone while preserving the pickaxe | Reach at least two stone while keeping pickaxe tier at least one | 2 |

Both labels passed the replay-backed endpoint and full-prefix preservation checks. Each retains the original failed attempt ID and hash, the actual requested `h`, the checked prefix and the earliest hit. These are observational offsets for `D_A`. The directed duration under `h_prime` remains unmeasured.

The versioned `hindsight/{sokoban,craftax}/D_A.jsonl` files retain the prior checked data and add one accepted train-support label per domain. The pool includes 24 original train-support candidates and one hindsight candidate per domain. Its v1 SHA-256 is `e451bd972eb4d5e888eee5c225cf59135955a6bbf1592662cff17913c79b3890`. The ledger binds this exact hash. The verify command rebuilt the pool and both datasets byte-for-byte. An isolated ledger rejected changing version 1 and accepted a distinct version 2. There were no Prompt 05 jobs at this freeze.

The artefacts are under `analysis/intent-pilot/hindsight/`. The `private/attempts/` and `private/replays/` directories hold six original attempts and their native checks. `private/fixtures/` holds the two integration traces, while `private/annotation_audit.jsonl` and `private/packet-index.jsonl` bind the blinded packets to the original failures. The candidate pool is `candidate-pool-v1.json` and its checksums are in `pool-manifest.json`. The attempt table and report are `attempts.csv` and `RESULTS.md`.

### Executed entry points and checks

```bash
source analysis/intent-pilot/env.sh
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase inference --activity hindsight-collect --seconds 700 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot hindsight collect
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase preparation --activity hindsight-packets --seconds 60 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot hindsight packets
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase preparation --activity hindsight-freeze --seconds 90 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot hindsight freeze
```

The annotation write took place between `packets` and `freeze`. The final focused tests passed 18 checks and the locked base `make test` passed 632 with four skips. The native Craftax checks ran in the shared collection worker. Ruff, Pyright, Mypy and `git diff --check` passed.

### Reconciled budget

The continuation charged **616.526 inference worker-seconds** and **401.541 preparation worker-seconds**. The single collection worker charged native setup and replay inside its envelope, with **253.135 s** itemized for native startup and staged roots. The two offline controllers used cached weights and metered USD was zero. All jobs settled. **3,245.717 worker-seconds remain**, with **1,445.717 available before Prompt 09** and **1,800 protected**. The job and annotation ledger export is `analysis/intent-pilot/hindsight/budget.json`.

### Exact resume command

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 4 --phase preparation --activity hindsight-verify --seconds 90 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot hindsight verify \
  > analysis/intent-pilot/stage05-hindsight-input.json
```

The saved output confirms the frozen pool, both checked dataset hashes, both accepted labels and the ledger binding.

### Next prompt — 05

Read the contract and journal and verify the frozen v1 pool. Use the frozen train-measurement split and actual controller revisions for directed requests. Check candidate applicability to each measurement context before rollout, predeclare the start snapshots and policy memory, and measure under the exact assigned `h` with shared budget accounting. Keep the observed hindsight offsets as training evidence. Store corrections and newly found candidates in a separately versioned pool.

## Prompt 05 — Directed measurements by controller and domain, 2026-11-22

The intake was `7000bdf` on `intents`. The measurement entry point extends `scripts.measure_intent_policy` and uses the shared active-intent executor and weighted midranks. The frozen measurement plan SHA-256 is `4d99b86508ebb50d7ec52d4563bb4fac02b6dbc0d85963ea280ac2f234c13256`. It binds the ledger-verified split, checked `D_A` hashes, and the original v1 support pool. The supplied Qwen3-0.6B and substitute Qwen2.5-1.5B-Instruct retain separate model, policy and dataset identities.

The plan selects three `train-measurement` contexts per domain from distinct canonical Sokoban layouts or Craftax crafting chains. The sampled groups are disjoint from policy support. Both models receive each context's same three typed requests. In Sokoban, the plan instantiates the frozen single-box, goal-count and two-box-conjunction types on short verified local witnesses. The Craftax context uses exact checked stone, sword and two-tool intents from the support pool. The evaluator uses the witnesses and native keys to lock the trials. Each policy query contains the public observation, public task, assigned predicates, instruction and bounded memory.

Two preparatory attempts ended before freezing the measurement plan. An archived Sokoban witness recorded moves past the first solved board, so selection now stops at that terminal board. The initial restriction to two newly filled goals within 12 actions was too narrow. The final selection requires a new occupied goal and a second checked box displacement for the conjunction, preferring a witnessed `RIGHT` when available. Both changes preceded any directed attempt. The initial failures are preserved in the ledger and freeze logs.

Every context begins with empty history, note and feedback for these two frozen policies. The fingerprint pins exact weight and tokenizer hashes, prompts, zero examples, greedy decoding with thinking disabled, the 64-token response cap, 1,024-token context cap, last-six memory, action parser, four-action batch and interrupt protocol. The attempt cap is 12 executed primitives. Sokoban plans one deterministic observation per candidate. Craftax plans two distinct native future keys with the same physical initial state and shares these keys across matching models and candidates. The cache reuses only exact greedy prompts within each policy fingerprint. Fixed counts are 9 planned Sokoban and 18 planned Craftax observations per model.

The whole 54-slot block used one 1,060-second reservation with a 1,020-second cooperative deadline. Trials are interleaved by context, candidate type, controller and domain. A malformed action response finishes the attempt as a failure immediately. Semantic timeout, task or episode termination without the intent, and failed or unknown preservation are eligible failures. The runner checkpoints requests and executed transitions before the next decision. Interrupted work and absent slots remain visible in the per-cell attempt file.

### Executed entry points

```bash
source analysis/intent-pilot/env.sh
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 5 --phase preparation --activity directed-freeze-contexts --seconds 85 -- \
  uv run --locked --offline --no-sync python -m scripts.measure_intent_policy pilot freeze
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 5 --phase inference --activity directed-native-block --seconds 1060 -- \
  uv run --locked --offline --no-sync python -m scripts.measure_intent_policy pilot run --wall-seconds 1020
```

### Measured coverage and scale

| Actual controller | Domain | Eligible / planned | Successes | Censored | Undispatched | Complete contexts | Informative contexts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | Sokoban | 5 / 9 | 2 | 0 | 4 | 1 | 0 |
| Qwen3-0.6B | Craftax | 8 / 18 | 0 | 1 | 9 | 1 | 0 |
| Qwen2.5-1.5B-Instruct | Sokoban | 4 / 9 | 0 | 0 | 5 | 1 | 0 |
| Qwen2.5-1.5B-Instruct | Craftax | 8 / 18 | 0 | 0 | 10 | 1 | 0 |

The reservation delivered 25 eligible observations. Qwen3 produced 11 semantic timeouts and two verified hits. Qwen2.5 produced 12 invalid action responses. One Qwen3 Craftax trial hit the campaign wall guard. Its partial trace remains in the attempt file, outside the denominator. The other 28 slots are undispatched. Each cell is explicitly incomplete, with one complete context and zero informative contexts.

An independent calculation read the actual attempts in the complete `cx-measure-01-0` context. Qwen3's sword request timed out on both native futures, so for it `p=0/2=0` and `d`, `w` and `u` are unknown. On the partial Sokoban context `gamma1_s2366`, Qwen3 reached the single-box and goal-count requests in one action each. Each has `p=1/1=1` and `d=1`. The third request is unmeasured, so weights and ranks remain masked. The two observed durations also coincide. The separate ties regression with durations `[2,5,5]` and probabilities `[1,.5,.5]` reproduces weights `[.5,.25,.25]` and midpoints `[.25,.75,.75]`.

The audit independently recomputed per-candidate counts and scaling with exact fractions and replayed a native Sokoban trace in each cell. Both controllers' repeated greedy payload checks passed. The Craftax receipts retain the distinct native future fingerprints and shared physical start. All 52 focused tests passed, covering the directed edge cases, the legacy policy measurement interface, active intent checkpoints, calibration and the existing LoRA data checks. Ruff, Mypy, Pyright and `git diff --check` passed.

### Outputs and budget

Each `analysis/intent-pilot/measurements/<controller>/<domain>/` contains `attempts.jsonl`, `pool.json`, `D_C.jsonl` and a checksum manifest. The pilot schema `carl-pilot-directed-v1` records eligible counts, successes, `p`, `d`, `w`, `u`, context completeness and uncertainty. The four score files have zero rows with `scale_supervision=true`. The shared report is `measurements/RESULTS.md`, four-cell summary `cells.csv`, and arithmetic proof `independent-check.json`. The full rollout and publication logs are `run.log` and `audit.log`.

Prompt 05 charged **1,031.056 inference worker-seconds** and **90.233 preparation worker-seconds**. The inference envelope includes **124.686 s** of native setup. All jobs settled with zero metered USD. **2,124.428 worker-seconds remain**, including **324.428 available before Prompt 09** and **1,800 protected**. The bounded block closes with the unfilled slots retained in the frozen pool. The settled export is `analysis/intent-pilot/measurements/budget.json`.

### Exact resume command

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.measure_intent_policy pilot report
uv run --locked --offline --no-sync python analysis/intent-pilot/measurements/independent_check.py
```

These are the two checked publication commands. They rebuild the reports from the frozen plan and private attempts. The measurements remain separately identifiable by controller and domain.

### Next prompt — 06

Read the contract and journal, then run the publication and audit commands. Keep the checked `D_A` lineage and exact `D_C` coverage for all four cells. Use only rows with informative measured scale for a relative arm. This block produced none, so carry that result into any matched comparison. Preserve the original pool, the remaining budget and the held-out partitions.

## Prompt 06 — Matched generators and validation, 2026-11-22

The intake was `73d51be` on `intents`. The stage bound all four generator profiles to the audited `D_C` cells, the original split and the corresponding frozen controller fingerprints. The only positive directed results were Qwen3-0.6B's two Sokoban successes. All four cells had zero rows with `scale_supervision=true`. Both requested relative scales are marked `no_informative_scale`. The three controller/domain cells without a reached example are marked `no_reached_examples` for the unlabelled run.

Both arms use one call with two slots, the same typed option menu, sampled support, weights, score and fixed slot-interleaved rollout scheduler. The relative arm exposes `u=0.25` and `u=0.75` in order with measured scale labels in the support. Every unlabelled example contains just its observation, task and typed outcome. Option order is hashed independently of requested scale. Semantic duplicate effects are recorded and evaluated once for a proposal call. A controlled native test exercises both wires, two distinct hit durations and the duplicate check.

Qwen3-0.6B's Sokoban diagnostic sampled two `D_A` examples at uniform weight and both reached `D_C` examples at observed `p=1`. Both measured examples lack comparative weights. The sampler records that it used observed probabilities for this diagnostic. The role has an empty generator history and shares one frozen weight instance with the policy. The policy only receives the current observation, task, assigned typed intent and its own action history. The actual policy keeps Prompt 05's 12-action cap and strict invalid-response handling.

The two fresh validation observations were from `generated_12x12x10_validation_3` and `generated_12x12x10_validation_0`. Each differs from the earlier direct and annotation observations. The menus use only visible move and push opportunities. The value is a ridge fit to `0.99 ** remaining_actions` from eight checked training trajectories and 429 states. It passed the layout and task-group holdout checks. Its fingerprint is `0ecd5899bab253e0be9877067cf20d2efd076cc8b9fee40a138ccd6e2c3e2a26`. Both arms use discounted endpoint value for verified hits and zero for completed failures. The configured Craftax fallback is a shared constant `V=0.5`. Neither validation run used that fallback.

### Measured validation and selection

| Diagnostic | Valid distinct / slots | Hits / eligible | First hits | G calls | G tokens in / out | Physical policy calls | Policy tokens in / out |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Initial instruction | 0 / 4 | — | — | 2 | 1,474 / 16 | 0 | 0 / 0 |
| Copy actual option IDs | 4 / 4 | 3 / 4 | 1, 1, 1 | 2 | 1,468 / 30 | 14 | 6,263 / 240 |

Both initial replies copied `ID1` and `ID2` from the format example. A second frozen edition removed the placeholders and requested a two-string JSON array of actual IDs. Both new replies chose two admissible effects. The original and revised user payloads, support, weights, context, decoder and option menus match exactly. Native replay verified the three first hits and the fourth attempt's 12-action timeout. There were zero duplicate effects within the two revised calls. Cross-arm overlap and success by requested scale remain unmeasured. The revised policy issued 15 logical calls, including one exact cache hit.

The validation selection binds the revised **unlabelled diagnostic** to config SHA-256 `456c8e25a01f3acbe5b05b8023aac7ed1691386496e3c84cedd19299516d4cbd`. Its manifest is `analysis/intent-pilot/generator-v2/selection.json`. The comparative status is `unavailable_no_informative_u`. The sealed test has no run in this stage. The measured prompt source is archived with its original hash. The final typed parser strips incidental prose from direct intent payloads before checking predicates. The audit replayed the measured prompts and proposals under the final parser and verified identical slots.

### Executed entry points

```bash
source analysis/intent-pilot/env.sh
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 6 --phase preparation --activity generator-freeze --seconds 45 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot generator freeze
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 6 --domain sokoban --phase inference --activity generator-validation --seconds 245 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot generator run --wall-seconds 225
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 6 --phase preparation --activity generator-copy-ids-freeze --seconds 35 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot generator freeze --edition 2
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 6 --domain sokoban --phase inference --activity generator-copy-ids-validation --seconds 225 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot generator run --edition 2 --wall-seconds 205
```

The first focused test exposed a prose normalization gap in duplicate detection. The corrected focused run passed all 23 checks. Ruff, Mypy, Pyright and `git diff --check` passed. The initial results and raw calls are in `analysis/intent-pilot/generator/`. The selected run, tables, value fit, per-slot results, raw generator calls, native rollouts and audit logs are in `analysis/intent-pilot/generator-v2/`.

### Reconciled budget

Prompt 06 charged **177.515 inference worker-seconds** and **80.879 preparation worker-seconds**, including the unsuccessful first instruction, the test correction and all audits. Every job settled with zero metered USD. **1,866.034 worker-seconds remain**. Of these, **66.034 are available before Prompt 09** and **1,800 are protected for it**. The export is `analysis/intent-pilot/generator-v2/budget.json`.

### Exact resume command

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot generator report --edition 2
uv run --locked --offline --no-sync python analysis/intent-pilot/generator-v2/audit.py
```

### Next prompt — 07

Read both documents and verify the frozen validation selection. Carry forward the unlabelled Sokoban diagnostic with the exact Qwen3 policy and fitted value identities. Treat policy-relative scale control as unmeasured and preserve the four `D_C` masks, sealed split and protected budget.

## Prompt 07 — Paired validation interrupted before the first response

### Frozen cases and eligibility

The only cell with separately observed elementary skills was Qwen3-0.6B in Sokoban. Its frozen policy walked to the assigned cell in one primitive on a Prompt 06 validation layout. In Prompt 05 it filled a new goal while preserving an occupied one in two separately assigned one-action attempts. The audit ties these observations to the same controller fingerprint.

| Controller | Domain | Prior directed hits / eligible trials | Prompt 07 status |
| --- | --- | --- | --- |
| Qwen3-0.6B | Sokoban | 2 / 5 | Two starts frozen for direct and unlabelled. Relative lacked informative `u`. |
| Qwen3-0.6B | Craftax | 0 / 8 | No qualifying composition under the frozen controller. |
| Qwen2.5-1.5B-Instruct | Sokoban | 0 / 4 | No qualifying composition under the frozen controller. |
| Qwen2.5-1.5B-Instruct | Craftax | 0 / 8 | No qualifying composition under the frozen controller. |

The evaluator froze `generated_12x12x10_validation_1` and `generated_12x12x10_validation_2` from different held-out layout groups. Neither layout was used for Prompt 06 generator validation. Each witness replays a walk followed by filling a new goal while preserving a specified occupied goal at all three observations. The model-facing decision used the starting observation, typed task and action vocabulary with empty initial memory. The witness and any future observations stayed private. Native Sokoban transitions and greedy decoding are deterministic. The sealed partition remained closed.

The paired plan SHA-256 is `098e2493611930eaca7ec8e55da71adbce034a4aa474e9afabba1c5b475daea9`. The private plan binds the frozen Prompt 06 selection, the same local policy and examples, and the train-only fitted value. The value estimates progress toward the whole-board goal and serves as a continuation heuristic for the partial task. The evaluator checks the actual typed partial goal separately. The common planned ceilings were 48 executed primitives, 48 native ticks, 16 logical requests, 16,000 charged tokens and 15 seconds per warmed episode. Probes share the action and request counters and may use at most half the count and token budgets. They leave a 0.75-second window for committing with an exact greedy cache hit and a fresh native transition. These caps were chosen under the remaining campaign budget. The depth-one execution path was never reached in this run.

### Observed results

The local CPU worker initialized the pinned FP32 checkpoint with six Torch threads. The runner recorded 38.827 seconds of setup, including 22.543 seconds loading the model and tokenizer. The campaign deadline interrupted the first direct generation. No response, native action, completed episode or paired comparison was observed. The saved request has 403 admitted input tokens and unknown output. The runner charged 467 tokens, including the reserved 64-token completion. The interrupted episode has unknown elapsed time and first-hit duration.

| Arm, Qwen3-0.6B / Sokoban | Solved / planned | Completed | First hit | All / live primitives | Logical / physical requests | Charged tokens | Episode seconds | Stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct | — / 2 | 0 | — | 0 / 0 | 1 / 1 | 467 | — | One campaign interruption, one undispatched |
| Depth-one unlabelled | — / 2 | 0 | — | — | — | — | — | Both undispatched |
| Depth-one relative | — / 2 | 0 | — | — | — | — | — | No informative scale support |

No full-search run was launched. The change adds `scripts.intent_pilot comparison` and a Sokoban diagnostic runner that restores probe snapshots and memory, accounts for G, V and policy work, and commits from the live parent. Ruff, Mypy, Pyright and `git diff --check` passed on the final source. The independent read-only audit passed for the frozen cases, prior skill evidence, first request, censored record and remaining budget. Model loading and admission of the first direct request were exercised. Depth-one with a real response awaits a runtime test.

### Executed entry points and budget

```bash
source analysis/intent-pilot/env.sh
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 7 --domain sokoban --phase preparation --activity comparison-freeze --seconds 5 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot comparison freeze
uv run --locked --offline --no-sync python -m scripts.intent_pilot run \
  --prompt 7 --domain sokoban --phase inference --activity paired-validation --seconds 59 -- \
  uv run --locked --offline --no-sync python -m scripts.intent_pilot comparison run --wall-seconds 55
```

The freeze-only invocation exited 124 after 5.063 worker-seconds. The combined worker froze the two cases before model loading and exited 124 after 59.334 worker-seconds. Its job charge includes cold startup and setup. The warm-up breakdown is recorded under `private/warmup.json` within that charge. The ledger settled both rows with USD 0 and no outstanding reservation. **1,801.637 worker-seconds remain**, of which **1.637 are available before Prompt 09** and **1,800 are protected**. The wrapper requires a minimum 2-second reservation including cleanup, so another experiment cannot start under the remaining early budget. Codex usage is recorded as unknown.

### Exact resume command

```bash
source analysis/intent-pilot/env.sh
uv run --locked --offline --no-sync python -m scripts.intent_pilot comparison report
uv run --locked --offline --no-sync python analysis/intent-pilot/comparison/audit.py
```

The report, CSV, JSON, archived executed source, frozen plan, last request and settled budget are in `analysis/intent-pilot/comparison/`. The resume commands read the saved run and verify its evidence without loading the model.

### Next prompt — 08

Read both campaign documents and the partial Prompt 07 report. The validation comparison has zero completed pairs. Full search remains gated on a completed validation comparison and sufficient campaign time. Relative generation also requires independently measured informative ranks. Preserve the sealed inputs and Prompt 09 reserve.
