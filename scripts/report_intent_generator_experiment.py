#!/usr/bin/env python3
"""Write a report from the available, recorded intent-experiment artefacts."""

import argparse
import fcntl
import json
import math
import os
import shlex
import time
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def rate(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "not measured"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def format_delta(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "not estimated"
    if abs(value) < 0.0005:
        return "0.000"
    return f"{value:+.3f}"


def format_interval(value: list[float] | None) -> str:
    if not value or len(value) != 2:
        return "not estimated"
    bounds = [0.0 if abs(bound) < 0.0005 else bound for bound in value]
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def gibibytes(value: object) -> str:
    return f"{value / 1024 ** 3:.2f} GiB" if isinstance(value, (int, float)) else "not measured"


def stage_option(state: dict, name: str, option: str) -> str | None:
    for stage in state.get("stages", []):
        if stage.get("name") != name:
            continue
        command = stage.get("command", [])
        if option in command:
            index = command.index(option)
            return str(command[index + 1]) if index + 1 < len(command) else None
    return None


def hardware_text(benchmark: dict, run_dir: Path) -> str:
    lines = []
    try:
        cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
        cpu = next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name"))
        lines.append(f"CPU: {cpu}.")
    except (OSError, StopIteration, IndexError):
        lines.append("CPU: not recorded.")
    env_log = run_dir / "logs/environment_probe.log"
    if env_log.exists():
        for line in env_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("python ", "torch ", "cuda ", "gpu ")):
                lines.append(line.strip() + ".")
    if benchmark:
        lines.append("Resource test: "
                     f"{benchmark.get('complete_step_seconds', 'not measured')} s per complete optimiser step; "
                     f"{benchmark.get('generation_tokens_per_second', 'not measured')} generation tokens/s; "
                     f"peak allocated/reserved GPU {gibibytes(benchmark.get('gpu_peak_allocated_bytes'))}/"
                     f"{gibibytes(benchmark.get('gpu_peak_reserved_bytes'))}; "
                     f"peak host RSS {gibibytes(benchmark.get('host_peak_rss_bytes'))}.")
        lengths = benchmark.get("token_lengths", {})
        if lengths:
            lines.append(f"Complete examples: {lengths.get('min')}–{lengths.get('max')} tokens "
                         f"(p95 {lengths.get('p95')}), limit {lengths.get('limit')}; "
                         f"truncated examples: {lengths.get('truncated')}.")
    return " ".join(lines)


def assessment_table(summary: dict, run_dir: Path) -> list[str]:
    results = summary.get("results", [])
    if not results:
        lines = ["The held-out assessment has not been completed."]
        for name, path in (("Base", run_dir / "base_outputs.jsonl"), ("Adapted", run_dir / "adapted_outputs.jsonl")):
            rows = read_jsonl(path)
            if rows:
                valid_json = sum(row.get("json_valid") is True for row in rows)
                lines.append(f"\n{name} generation has {len(rows)} saved outputs; "
                             f"strict JSON valid: {rate(valid_json, len(rows))}. "
                             "Non-triviality and feasibility await assessment.")
        return lines
    lines = [
        "| Method | JSON valid | Valid non-trivial | Path found by bounded search | Unresolved among valid | Missing outputs |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for item in results:
        outputs = item.get("outputs")
        valid = item.get("valid_nontrivial")
        lines.append(f"| {item.get('method', 'unknown')} | {rate(item.get('valid_json'), outputs)} "
                     f"| {rate(valid, outputs)} | {rate(item.get('witness_found'), outputs)} "
                     f"| {rate(item.get('unresolved_among_valid'), valid)} "
                     f"| {item.get('missing_infrastructure_outputs', 'not recorded')} |")
    solver = summary.get("solver", {})
    if solver:
        lines.append(
            f"\nThe search limit was depth {solver.get('max_depth')}, {solver.get('max_nodes')} expanded nodes "
            f"and {solver.get('seconds')} s per proposal. Each action sequence found by search was replayed "
            "in the environment and reached a state satisfying the proposed intent. These results do not "
            "establish full-puzzle success or controller performance; a cutoff is unresolved.")
    paired = summary.get("paired_adapted_minus_base", {})
    if paired.get("contexts"):
        lines.append(f"\nPaired adapted-minus-base valid-intent difference: {format_delta(paired.get('valid_delta'))}; "
                     f"95% board bootstrap {format_interval(paired.get('valid_delta_95_board_bootstrap'))}; "
                     f"{paired['contexts']} paired contexts across {paired.get('boards')} boards.")
        lines.append(
            f"\nPaired adapted-minus-base difference in the fraction reached by bounded search: "
            f"{format_delta(paired.get('witness_delta'))}; "
            f"95% board bootstrap {format_interval(paired.get('witness_delta_95_board_bootstrap'))}.")
    for item in results:
        if item.get("outputs"):
            statuses = {
                ("path found" if name == "witness_found" else name.replace("_", " ")): count
                for name, count in item.get("solver_statuses", {}).items()
            }
            lines.append(f"\n{item.get('method')}: effect predicates {item.get('effect_types', {})}; "
                         f"conjunction sizes {item.get('conjunction_sizes', {})}; "
                         f"search outcomes {statuses}; "
                         f"{item.get('solver_expanded_nodes', 0)} nodes and {item.get('solver_seconds', 0):.2f} s "
                         "of bounded-search time.")
    return lines


def posthoc_intent_composition(run_dir: Path) -> list[str]:
    rows = read_jsonl(run_dir / "fence_feasibility/feasibility.jsonl")
    valid = {
        method: [row for row in rows if row.get("method") == method and row.get("valid_nontrivial") is True]
        for method in ("base", "adapted")
    }
    lines = []
    thresholds: Counter[int] = Counter()
    for row in valid["base"]:
        effects = (row.get("intent") or {}).get("effects", [])
        thresholds.update({
            effect["count"]
            for effect in effects
            if effect.get("predicate") == "boxes_on_goals_at_least" and isinstance(effect.get("count"), int)
        })
    if thresholds:
        threshold, count = thresholds.most_common(1)[0]
        full_puzzle = all(
            sum(tile in "$*"
                for line in row.get("start_observation", [])
                for tile in line) == threshold
            for row in valid["base"]
            if any(
                effect.get("predicate") == "boxes_on_goals_at_least" and effect.get("count") == threshold
                for effect in (row.get("intent") or {}).get("effects", [])))
        suffix = ", which asks for every box on a goal on these boards" if full_puzzle else ""
        lines.append(f"Among valid base proposals, {count}/{len(valid['base'])} included "
                     f"`boxes_on_goals_at_least({threshold})`{suffix}.")
    box_targets: Counter[tuple[int, int]] = Counter()
    box_proposals = 0
    for row in valid["adapted"]:
        effects = (row.get("intent") or {}).get("effects", [])
        targets = {(effect["row"], effect["col"]) for effect in effects if effect.get("predicate") == "box_at"}
        box_proposals += bool(targets)
        box_targets.update(targets)
    if valid["adapted"]:
        lines.append(f"Among valid adapted proposals, {box_proposals}/{len(valid['adapted'])} included `box_at`.")
    if box_targets:
        (target_row, target_col), count = box_targets.most_common(1)[0]
        lines.append(
            f"The most common adapted target was `box_at({target_row}, {target_col})` in "
            f"{count}/{len(valid['adapted'])} valid contexts; contexts from the same board are not independent.")
    if valid["base"] and valid["adapted"]:
        lines.append("Each generator is evaluated on its own proposals; effect predicates and difficulty "
                     "are not matched. A difference in the fraction reached can therefore reflect the difficulty "
                     "of the proposed effects as well as generator quality.")
    return lines


def examples(run_dir: Path) -> list[str]:
    rows = read_jsonl(run_dir / "assessment/assessment.jsonl")
    picked = []
    for method in ("base", "adapted", "local_rule"):
        candidates = [row for row in rows if row.get("method") == method and row.get("status") == "evaluated"]
        if not candidates:
            continue
        selected = next((row for row in candidates if row.get("valid_nontrivial")), candidates[0])
        picked.append(selected)
    if not picked:
        return ["Examples will be added when the held-out assessment is available."]
    lines = []
    for row in picked:
        lines.extend([
            f"### {row['method']} — `{row['record_id']}`", "", "Initial board:", "```text",
            *row.get("start_observation", []), "```", "", "Raw proposal:", "````text",
            str(row.get("raw_output", "")), "````", "",
            f"Validity: {row.get('valid_nontrivial')}; reasons: {row.get('validation_errors', [])}."
        ])
        solver = row.get("solver") or {}
        if solver.get("witness_found"):
            lines.extend([
                f"Bounded search found a sequence of {solver.get('witness_depth')} primitive actions "
                "reaching a state satisfying the proposed intent:", "```text",
                *solver.get("reached_observation", []), "```", ""
            ])
        elif row.get("demonstration_later_observation"):
            lines.extend([
                "Recorded later demonstration state (not a controller result):", "```text",
                *row["demonstration_later_observation"], "```", ""
            ])
    return lines


def build_report(run_dir: Path, state: dict) -> str:
    data = read_json(run_dir / "dataset_manifest.json")
    benchmark = read_json(run_dir / "qwen3_benchmark.json") or read_json(run_dir / "benchmark.json")
    status = read_json(run_dir / "adapter/status.json")
    best = read_json(run_dir / "adapter/best.json")
    curve = read_jsonl(run_dir / "adapter/training_curve.jsonl")
    summary = read_json(run_dir / "assessment/summary.json")
    frozen = read_json(run_dir / "frozen_evaluation_contexts.json")
    records = read_jsonl(run_dir / "dataset.jsonl")
    provenance = Counter(row.get("annotation_provenance", {}).get("kind", "unknown") for row in records)
    start = state.get("clock", {}).get("start_utc", "not recorded")
    deadline = state.get("clock", {}).get("deadline_utc", "not recorded")
    elapsed = state.get("elapsed_seconds")
    if elapsed is None and state.get("clock", {}).get("start_monotonic") is not None:
        elapsed = max(0, time.monotonic() - state["clock"]["start_monotonic"])
    elapsed_text = f"{elapsed / 60:.1f} min" if isinstance(elapsed, (int, float)) else "not recorded"
    lines = [
        "# Sokoban intent generator experiment", "", f"Run status: **{state.get('status', 'unknown')}**.",
        f"Start: {start}; hard deadline: {deadline}; elapsed at this report: {elapsed_text}.", "", "## Data audit", ""
    ]
    if data:
        lines.extend([
            f"Family: {data.get('family', 'unknown')}; {data.get('classification', '')}.", "",
            "| Split | Independent boards | Checked fragments | Effect types |", "|---|---:|---:|---|"
        ])
        for split in ("train", "validation", "test"):
            item = data.get("counts", {}).get(split, {})
            lines.append(
                f"| {split} | {item.get('boards', 'unknown')} | {item.get('records', 'unknown')} | {item.get('effect_types', {})} |"
            )
        lines.append("\n`walk_only` describes a player-position effect; the corresponding recorded "
                     "fragment may also contain pushes.")
        trajectories = data.get("trajectories", [])
        if trajectories:
            primitive = sum(item.get("primitive_actions", 0) for item in trajectories)
            pushes = sum(item.get("push_actions", 0) for item in trajectories)
            walks = sum(item.get("walking_actions", 0) for item in trajectories)
            noops = sum(item.get("no_ops", 0) for item in trajectories)
            suffixes = sum(item.get("discarded_post_terminal_actions", 0) for item in trajectories)
            checks = sum(item.get("historically_recorded_states_checked", 0) for item in trajectories)
            lines.append(f"\nEnvironment replay: {primitive} primitive actions ({pushes} pushes, {walks} walks), "
                         f"{noops} no-ops, {checks} checks against recorded states. "
                         f"{suffixes} post-terminal actions were excluded and recorded separately.")
        lines.extend([
            f"\nAnnotation provenance: {dict(provenance)}. Historical shared-context Terra labels "
            "are distinguished from fresh isolated queries; prior Craftax records are outside this domain.",
            f"\nThe held-out selection contains {frozen.get('count', 'unknown')} contexts across "
            f"{frozen.get('independent_boards', 'unknown')} boards. Overlapping contexts are not "
            "independent boards. Layouts and their symmetries were separated before annotation.",
            f"\nDataset SHA-256: `{data.get('dataset_sha256', 'not recorded')}`. "
            "Source paths, hashes, trajectory lengths, split and exposure audit are in `dataset_manifest.json`."
        ])
    else:
        lines.append("The data preparation stage has not completed; see the stage log and `rejected.jsonl` if present.")
    if benchmark and benchmark.get("token_lengths", {}).get("n"):
        length_audits = "The resource benchmark and training setup" if status else "The resource benchmark"
        lines.append(f"\n{length_audits} tokenised complete examples from all "
                     "splits, including held-out target labels, solely to audit sequence lengths. "
                     "Those test labels were not used for a model forward pass, loss, optimisation or "
                     "checkpoint selection; the generator did not receive future observations.")
    lines.extend(["", "## Annotation and controller", ""])
    terra_summaries = list(run_dir.glob("terra*/batch_*_summary.json"))
    accepted = sum(len(read_json(path).get("accepted", [])) for path in terra_summaries)
    lines.append(f"Fresh isolated Codex annotations accepted: {accepted}; "
                 f"recorded batch summaries: {len(terra_summaries)}. "
                 "Raw event streams and errors are retained in the annotation directories.")
    first_trace = run_dir / "terra/batch_0000_attempt_1/events.jsonl"
    if first_trace.exists():
        trace = first_trace.read_text(encoding="utf-8", errors="replace")
        if "not supported when using Codex with a ChatGPT account" in trace:
            lines.append("\nThe initial CLI pilot used the saved ChatGPT session and received "
                         "`model_not_found` (HTTP 400) for `gpt-6.1-terra` on that authentication path. "
                         "This does not establish unavailability through other channels.")
    review = read_json(run_dir / "decision_notes.json").get("terra_api_review", {})
    if review.get("result") == "rejected_before_execution":
        lines.append("\nThe automatic reviewer rejected the proposed single retry using the existing API key "
                     "before execution because it could transmit a local board pair externally and incur a "
                     "charge without explicit authorisation for that route. No request was made by that attempt.")
    lines.append("\nNo verified, inexpensive fixed predicate-conditioned controller was run. "
                 "The repository also contains a remote text-goal prototype; it was not measured here. "
                 "No policy-relative scale labels were inferred from demonstration offsets.")
    lines.append("\nThe local rule baseline uses only the current board. It computes where the player can walk "
                 "without moving boxes, selects a legal push whose standing position is reachable, and prefers "
                 "a destination on a goal. If no push is available it proposes a reachable player position "
                 "when one exists. This baseline explicitly uses local feasibility information.")
    lines.extend(["", "## Model and training", "", hardware_text(benchmark, run_dir), ""])
    if benchmark:
        lines.append(f"Model: `{benchmark.get('model')}`, revision `{benchmark.get('revision')}`; "
                     f"dtype {benchmark.get('dtype')}, LoRA rank {benchmark.get('rank')}, "
                     f"gradient accumulation {benchmark.get('accumulation')}, "
                     f"thinking enabled: {benchmark.get('enable_thinking')}.")
    base_summary = read_json(run_dir / "base_outputs.summary.json")
    missing_base = [key for key in ("prompt_sha256", "dtype", "max_length") if key not in base_summary]
    if base_summary and missing_base:
        length_option = stage_option(state, "base_evaluation", "--max-length")
        dtype_option = stage_option(state, "base_evaluation", "--dtype")
        lines.append("\nThe base-generation summary did not record " + ", ".join(f"`{key}`" for key in missing_base) +
                     f". The recorded stage command specified max length {length_option or 'not specified'} "
                     f"and dtype {dtype_option or 'not specified (the then-current default applied)'}; "
                     f"the resource benchmark recorded dtype {benchmark.get('dtype', 'unknown')}. "
                     "The base summary alone cannot verify the original prompt byte-for-byte.")
    if status:
        lines.append(f"\nTraining status: **{status.get('status', 'unknown')}**, "
                     f"completed steps: {status.get('completed_steps', 'unknown')}. "
                     f"Best checkpoint: `{best.get('adapter', status.get('best_adapter'))}`; "
                     f"validation loss: {best.get('validation_loss', 'not recorded')}.")
        lines.append("\nFull training peak GPU allocated/reserved according to Torch: "
                     f"{gibibytes(status.get('gpu_peak_allocated_bytes'))}/"
                     f"{gibibytes(status.get('gpu_peak_reserved_bytes'))}. Device-wide figures from "
                     "`nvidia-smi` may differ because they include other allocations and driver overhead.")
        if curve:
            lines.append(f"\nTraining loss first → last: {curve[0].get('train_loss')} → "
                         f"{curve[-1].get('train_loss')}; this is not the primary result.")
    else:
        lines.append("\nNo completed training status is available; do not infer that an adapter was trained.")
    for method, filename in (("Base", "base_outputs.summary.json"), ("Adapted", "adapted_outputs.summary.json")):
        generation = read_json(run_dir / filename)
        elapsed_generation = generation.get("elapsed_seconds")
        if isinstance(elapsed_generation, (int, float)):
            restored = generation.get("resumed_records")
            resume_note = f"; {restored} records were restored before this invocation" if restored else ""
            lines.append(f"\n{method} generation loop: {elapsed_generation:.1f} s "
                         f"for {generation.get('n', 'unknown')} saved outputs{resume_note}; "
                         "model loading is excluded from this elapsed time.")
    lines.extend(["", "## Held-out comparison", ""])
    lines.extend(assessment_table(summary, run_dir))
    fence_audit = read_json(run_dir / "fence_audit/fence_audit_summary.json")
    if fence_audit.get("results"):
        lines.extend([
            "", "### Exploratory formatting sensitivity", "",
            "After seeing the raw outputs, one outer Markdown code fence was removed symmetrically "
            "from both models' outputs. This post-hoc check does not change the strict primary result; "
            "this format audit itself only checks validity.", "",
            "| Method | JSON valid after fence removal | Valid non-trivial after fence removal | Fences removed |",
            "|---|---:|---:|---:|"
        ])
        for item in fence_audit["results"]:
            lines.append(f"| {item.get('method')} | {rate(item.get('posthoc_valid_json'), item.get('outputs'))} "
                         f"| {rate(item.get('posthoc_valid_nontrivial'), item.get('outputs'))} "
                         f"| {item.get('single_outer_fence_removed', 0)} |")
    fence_feasibility = read_json(run_dir / "fence_feasibility/summary.json")
    if fence_feasibility.get("results"):
        lines.extend([
            "", "### Exploratory feasibility after fence removal", "",
            "The same symmetric post-hoc format normalisation was followed by a separate bounded "
            "search. These numbers are secondary and were not used for training or checkpoint selection.", ""
        ])
        lines.extend(assessment_table(fence_feasibility, run_dir))
        lines.extend("\n" + line for line in posthoc_intent_composition(run_dir))
        if (run_dir / "fence_feasibility/examples.md").exists():
            lines.append("\nNormalised examples and their action sequences found by bounded search are in `fence_feasibility/examples.md`.")
    figures = [path.name for path in sorted((run_dir / "figures").glob("*"))] if (run_dir / "figures").exists() else []
    lines.append(f"\nFigure files: {', '.join(figures) if figures else 'not yet available'}. "
                 "The processed JSON/CSV and individual proposals retain the exact denominators and cutoffs.")
    lines.extend(["", "## Example inspection", ""])
    if (run_dir / "assessment/examples.md").exists():
        lines.append(
            "Full selected examples, predicate truth values and action sequences found by bounded search are in `assessment/examples.md`.\n"
        )
    lines.extend(examples(run_dir))
    lines.extend([
        "", "## Limitations and proposal text", "",
        "The layouts come from a previously selected synthetic exploratory cohort; prior exposure "
        "cannot be ruled out. Three independent validation boards provide limited checkpoint-selection evidence. "
        "Predicate validity and paths found by bounded search do not establish transfer, full-puzzle success "
        "or policy-relative reachability.", ""
    ])
    lines.append("The current supervisor starts stages under a persistent session leader configured as a Linux "
                 "subreaper, and the watchdog tracks observed members with pidfds; separate annotation sessions "
                 "are registered. Earlier stages may have used an older supervisor. There is no cgroup boundary: "
                 "if the leader is forcibly killed while a new child is created and orphaned between scans, "
                 "before its pidfd is acquired, that child could escape tracking. A cgroup would provide stronger "
                 "containment.")
    lines.append("")
    by_method = {row.get("method"): row for row in summary.get("results", [])}
    base, adapted = by_method.get("base", {}), by_method.get("adapted", {})
    if base.get("outputs") and adapted.get("outputs"):
        paragraph = ("> In an exploratory Sokoban study, a small pretrained model was adapted to propose structured "
                     "effects from the current board. Under strict JSON parsing on held-out contexts, valid "
                     "non-trivial intents occurred in "
                     f"{base.get('valid_nontrivial')}/{base.get('outputs')} base proposals and "
                     f"{adapted.get('valid_nontrivial')}/{adapted.get('outputs')} adapted proposals. ")
        normalised = {row.get("method"): row for row in fence_feasibility.get("results", [])}
        before, after = normalised.get("base", {}), normalised.get("adapted", {})
        if before.get("outputs") and after.get("outputs"):
            paragraph += ("In a post-hoc analysis removing a single outer code fence, the corresponding "
                          f"valid counts were {before['valid_nontrivial']}/{before['outputs']} and "
                          f"{after['valid_nontrivial']}/{after['outputs']}; bounded search found paths "
                          f"for {before['witness_found']}/{before['outputs']} and "
                          f"{after['witness_found']}/{after['outputs']}, respectively. ")
            paragraph += "Proposals were not matched for effect type or difficulty. "
        paragraph += "These results do not establish controller reachability or transfer."
        lines.append(paragraph)
    else:
        lines.append("No quantitative paragraph for the proposal is supported until both comparisons are available.")
    lines.extend([
        "", "## Reproduction", "", "The run manifest records the exact stage commands, timeouts and output locations. "
        "For a new run, use `uv run --no-sync python scripts/run_intent_generator_experiment.py "
        "--run-dir analysis/<new-run>`; the entry point creates its own watchdog and never resets "
        "an existing run's deadline. After interruption, add `--resume` to the same command and "
        "point `--run-dir` at the existing directory. It waits for registered children, preserves "
        "partial results and refuses expensive stages after the original cutoff.", ""
    ])
    for stage in state.get("stages", []):
        name = stage.get("name", "")
        if any(word in name for word in ("base_evaluation", "train", "adapted_evaluation", "assessment")):
            command = stage.get("command", [])
            if command:
                lines.extend([f"**{name}** ({stage.get('status')}):", "```bash", shlex.join(command), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def refresh_report_unlocked(run_dir: Path, state: dict | None = None) -> None:
    """Replace the report; the caller must already hold the supervisor lock."""
    state = state if state is not None else read_json(run_dir / "run_manifest.json")
    report = run_dir / "REPORT.md"
    old = report.read_text(encoding="utf-8") if report.exists() else ""
    begin, end = "<!-- supervisor status begin -->", "<!-- supervisor status end -->"
    supervisor_section = ""
    if begin in old and end in old:
        supervisor_section = "\n\n" + begin + old.split(begin, 1)[1].split(end, 1)[0] + end + "\n"
    text = build_report(run_dir, state) + supervisor_section
    temp = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, report)


def write_report(run_dir: Path) -> None:
    """Serialise report updates with the watchdog and replace atomically."""
    run_dir = run_dir.resolve()
    with (run_dir / ".supervisor.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            refresh_report_unlocked(run_dir)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    write_report(args.run_dir)


if __name__ == "__main__":
    main()
