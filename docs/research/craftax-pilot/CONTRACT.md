# Sokoban and Craftax intent pilot contract

This campaign uses native Sokoban and Craftax transitions. Codex is the offline annotator for both domains. The two frozen controller arms are Qwen3-0.6B and Qwen3-1.7B. Run both from verified local checkpoints. Keep the base revision, tokenizer, prompt, decoding, action parser, memory protocol and device settings in the controller fingerprint. Report each arm separately.

The local proposal's Method and Evaluation sections guide the pilot. The starting checkout is `intents` at `5ddd4cb7bcde591fdcf7a0f32b0aa41b460e8b0c`. The checkout inspected at `2a60ff91b5767a06dfff83166e91d80afad3a13a` and the earlier queue at `313a57c403ce9531aea321155bb85fa228d42d65` are points in its history. [Status and resume instructions](STATUS.md)

## Splits and access

| Partition | Use |
| --- | --- |
| Train support | Replay-checked demonstrations, Codex annotation and example support for π and G |
| Train measurement | Disjoint contexts and intent pools for restored controller trials |
| Validation | Prompt, example-selection and budget choices on separate episodes |
| Sealed test | Final evaluation of frozen settings on whole held-out task compositions |

Freeze membership before extracting any fragments, transforming any boards or asking Codex for labels. `scripts.intent_pilot freeze` hashes the complete split files and binds the resulting manifest to the campaign ledger. An existing split file must match its frozen content. A changed corpus or split requires a new campaign decision.

Sokoban groups include the original trajectory ID, the declared task family and CaRL's D4-canonical walls-and-goals layout. All rotations, reflections and duplicate layouts share a group. The implementation rejects cross-partition aliases and retains one witness per canonical layout. The four cohort families are the real 12×12 four-box demonstrations, 12×12 ten-box gamma1 witnesses, 12×12 ten-box generated validation witnesses and a seeded sample of the 12×12 ten-box gamma2 family. The remainder of the gamma2 pool is outside the pilot. Its composition family is reserved for sealed evaluation throughout the campaign. The broad family shift, and the support difference in box count, belong in any interpretation of transfer.

Craftax freezes planned native episodes with map size 16 and a 32-step environment horizon. The pure split step pins each world by its generator, map parameters and derived reset seed. Each canonical crafting chain stays in one partition. Chain identity follows milestone order, resource types and tool tiers. It normalizes count magnitudes, conjunction order and repeated adjacent milestones. Use each source episode ID and world once. The sealed compositions use individual skills present in training. Confirm native resets, actual map identities, rewards and restorable snapshots during collection, before extracting any fragments. The planned worlds do not supply observed outcomes.

The bundled Sokoban quick-start layouts, the previously showcased Sokoban source, and the documented five-action Craftax recipe are fixtures. Exclude their relevant canonical layouts or whole chain from the campaign. The Craftax fixture reset seeds are also excluded. Fixture replay can check integration separately with a recorded cost.

The split seed is `2026112201` for choosing solved Sokoban sources, `2026112202` for Craftax worlds, and `2026112203` for subsequent fragment extraction. Preserve the original source hashes and membership hashes. Keep the private roster and sealed files under `analysis/intent-pilot/`. Only the evaluator reads sealed traces or reference chains. Prompt and training code uses the role-specific unsealed inputs shown by `verify`. A sealed model query contains its public goal and live observations and action history. File paths, split metadata, future observations, traces, hidden state, witness actions and solution chains stay outside the model input.

## Annotation and evaluation

For a training fragment, show Codex its start history, observed continuation, public task goal and domain schema. Codex proposes structured effects and preservation conditions or abstains. Use simulator replay and typed predicates for acceptance. Require an initially unmet intent, evaluate preservation at the start and every step until the first checked effect hit, and retain rejection reasons and unknown facts. Keep raw observations and annotation evidence separate from the generator's model-facing rows. Demonstration labels in `D_A` have missing scale. Preserve Codex provenance and all attempted annotations.

For controller measurements, freeze the candidate types and IDs before sampling. Start every attempt from the same simulator snapshot and initial controller memory, with deterministic independent seeds, a fixed action cap and an unchanged controller. Retain first-hit duration, task success, violations, unknowns, terminal setting and cost separately. Use the weighted midrank of successful durations in each context for the policy-relative (u), with weights proportional to the measured success fractions. Keep failed and incomplete attempts in the audit. Compare demonstration-only, measured without scale and measured with scale using matched pairs, resources and tuning budgets.

Report task success and intent progress separately. Count all simulator decisions and model calls, including warm-up, invalid/no-op actions, speculative branches, failures and retries. Report preparation and inference costs separately. Checkpointing, prompt selection and stopping decisions use the declared validation split. Open the sealed group only for the matched final evaluation. Treat the public integration fixtures as plumbing results.

## Budget and handoff

Use one SQLite ledger at `analysis/intent-pilot/campaign.sqlite3`. The ceiling is USD 0 for metered services and 7,200 cumulative local worker-seconds. Reserve 1,800 for Prompt 09. Prompts 01–08 can commit up to 5,400 worker-seconds. Time from concurrent jobs is added. For one job with internal concurrency, declare the maximum number of independent concurrent workers. The wrapper charges its elapsed time for every declared worker. CPU math threads belong to a worker and should be recorded as hardware settings.

Budget each experiment with `scripts.intent_pilot run`. Set `--phase preparation` for data work, warm-up and integration before the measured trial, and `--phase inference` for measured controller execution. Record controller calibration as an inference activity with a distinct name from final evaluation. The ledger reserves the timeout plus a second per worker for startup and cleanup, enforces the command deadline with GNU `timeout`, and charges actual elapsed time. Failed launches, retries and completed timeouts remain separate rows. An unsettled reservation stays committed after an interruption. Check the ledger before issuing a retry.

Keep model downloads, human annotation, Codex annotation and observable Codex usage in separate preparation rows via `record-preparation`. Omit unavailable fields so they appear as null. Run local commands only. There is no Luna, Sol or Astra inference, and no metered provider fallback. A missing weight or incompatible runtime is a preparation dependency.

Every stage begins by reading this file and [STATUS.md](STATUS.md). After doing the work, append the exact tested resume command, frozen inputs, output paths, results and failure counts, remaining budget, and the next prompt. Put generated traces, files, caches and logs under `analysis/intent-pilot/`. The Markdown journal records each handoff. The SQLite ledger supplies the current budget.
