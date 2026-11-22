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
