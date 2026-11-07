"""Export a measured generator checkpoint and inspectable prompts without inference."""
import argparse
import hashlib
import json
from pathlib import Path

from carl.icl.boards import from_rows
from carl.icl.checkpoints import GeneratorCheckpoint, export_generator_checkpoint, load_generator_checkpoint
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment, load_data
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from carl.icl.types import JSONObject
from scripts.generator_probe.calibrate import aggregate, digest
from scripts.generator_probe.integrity import verify_target
from scripts.generator_probe.prepare_dataset import write_once
from scripts.luna_generator_probe import BASE, OUTPUT, PROMPT, configurations

DEFAULT_CALIBRATION = OUTPUT / "expanded_calibration/openai/calibration.json"
DEFAULT_OUTPUT = OUTPUT / "checkpoint/generator.json"


def file_digest(path: Path) -> str:
    """Hash source bytes without displaying their contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(calibration_path: Path, calibration: JSONObject) -> dict[str, str]:
    """Bind every target's raw trials and all frozen calibration protocol inputs."""
    manifest_path = calibration_path.parent / "manifest.json"
    manifest: JSONObject = json.loads(manifest_path.read_text())
    if digest(manifest) != calibration["signature"]:
        raise ValueError("Calibration signature differs from its prepared manifest")
    if manifest["provider"] != "openai" or calibration.get("status") != "completed":
        raise ValueError("Export requires completed live calibration")
    paths = {Path(key) if Path(key).is_absolute() else ROOT / key for key in manifest["hashes"]}
    for key, expected in manifest["hashes"].items():
        path = Path(key) if Path(key).is_absolute() else ROOT / key
        if file_digest(path) != expected:
            raise ValueError(f"Prepared calibration source changed: {key}")
    dataset_path = OUTPUT / "training_candidates.json"
    dataset: JSONObject = json.loads(dataset_path.read_text())
    rows: list[JSONObject] = []
    for context in dataset["contexts"]:
        for index, _candidate in enumerate(context["candidates"]):
            path = calibration_path.parent / f"{context['id']}_{index}.json"
            row: JSONObject = json.loads(path.read_text())
            verify_target(context, index, row, manifest["policy_fingerprint"])
            rows.append(row)
            paths.add(path)
    if len(rows) != manifest["targets"] or len(rows) != 30:
        raise ValueError("Expected evidence for all thirty candidate targets")
    rebuilt = aggregate(dataset["contexts"], rows, manifest)
    for field in ("status", "result_digest", "examples", "targets", "contexts", "selected_example_ids"):
        if rebuilt[field] != calibration[field]:
            raise ValueError(f"Calibration summary differs from raw target evidence: {field}")
    paths.update({calibration_path, manifest_path, dataset_path, BASE, PROMPT,
                  Path(__file__).resolve(), ROOT / "scripts/luna_generator_probe.py",
                  OUTPUT / "validation_sources_30.json", OUTPUT / "dataset_selection_exclusions.json",
                  OUTPUT / "dataset_provenance_checks.json"})
    config = configurations()["planned_board_low"]
    paths.add(config.path("data_dir") / "training.json")
    if config.value_examples == "trajectory":
        paths.add(config.path("data_dir") / "value_examples.json")
    return {str(path.resolve()): file_digest(path) for path in sorted(paths)}


def prompt_bundle(config: ExperimentConfig, checkpoint: GeneratorCheckpoint) -> JSONObject:
    """Render full requests without invoking generation, policy, value, or embeddings."""
    training, _instances = load_data(config)
    sources: JSONObject = json.loads((OUTPUT / "validation_sources_30.json").read_text())
    source = sources["instances"][0]
    state = from_rows(source["state"])
    payloads: list[JSONObject] = []
    limits = list(dict.fromkeys([min(4, checkpoint.actual_example_count), checkpoint.actual_example_count]))
    for limit in limits:
        condition = config.model_copy(update={"generator_example_limit": limit})
        api = ExperimentAPI(condition.path("cache_dir"), "replay", condition.limits)
        components = ICLComponents(condition, api, environment(state), training)
        for scale in (0.5, 0.85):
            payloads.append({"source_id": source["id"], "example_count": limit, "u": scale,
                             "payload": components.generator_payload(state, scale, resolve_embeddings=False)})
        if api.metrics["logical_requests"]:
            raise ValueError("Prompt export unexpectedly attempted inference")
    context = training["contexts"][0]
    train_state = from_rows(context["state"])
    train_target = from_rows(context["candidates"][0]["subgoal"])
    api = ExperimentAPI(config.path("cache_dir"), "replay", config.limits)
    components = ICLComponents(config, api, environment(train_state), training)
    return {"checkpoint_sha256": checkpoint.sha256, "source": source, "generator": payloads,
            "policy": {"query_context": context["id"], "query_kind": "training witness inspection only",
                       "payload": components.policy_payload(train_state, train_target)},
            "value": {"query_context": context["id"], "query_kind": "training source inspection only",
                      "payload": components.value_payload(train_state)},
            "inference_calls": 0}


def inspection_markdown(checkpoint: GeneratorCheckpoint, config: ExperimentConfig) -> str:
    """Show ordered labels, exact demonstration boards, and all prompt instructions."""
    lines = ["# Calibrated generator checkpoint", "",
             f"Retained **{checkpoint.actual_example_count} successful demonstrations** from 30 candidate targets "
             "on 10 training layouts. Zero-hit targets remain in calibration evidence and are excluded from examples.",
             "", "Calibration used R=2 independently seeded categorical action streams, conditional on cached "
             "policy distributions, with L=8. Labels describe the three selected targets per context; they are "
             "not percentiles over all possible future states. This is exploratory calibration.", "",
             "Validation contains 30 source states on 10 distinct layouts; positions on one layout are dependent. "
             "No validation action witnesses or future target boards appear in the generator prompts.", "",
             f"Checkpoint SHA-256: `{checkpoint.sha256}`", "",
             f"Frozen policy fingerprint: `{checkpoint.policy_fingerprint}`", "",
             f"Model: `{checkpoint.model}`. Generator reasoning: `{checkpoint.defaults.reasoning_effort}`. "
             f"Policy few-shot contexts: {checkpoint.policy_few_shot}.", "",
             "`generator_inputs.json` contains complete generator requests at u=0.5 and u=0.85 with 4/all "
             "examples on one validation source, plus complete policy/value requests using training queries only. "
             "`config.json` contains effective settings. These requests were rendered without inference.", "",
             "The paired 30-source comparison separately saves its complete 60 scheduled requests. "
             "Reachability alone does not establish adherence to the requested scale.", "",
             "| Order | Context | Target | k | Mean first hit d | Hits/R | Weight | u | a_c |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for number, row in enumerate(checkpoint.examples, 1):
        lines.append(f"| {number} | {row.context} | {row.target_index} | {row.k} | {row.d:g} | "
                     f"{row.hits}/{row.trials} | {row.weight:.6g} | {row.u:.6g} | {row.a_c:g} |")
    lines.extend(["", "## Full prompt instructions", ""])
    for name, prompt in config.prompts.model_dump().items():
        lines.extend([f"### {name}", "", "```text", str(prompt), "```", ""])
    lines.extend(["## Ordered generator demonstrations", ""])
    for number, row in enumerate(checkpoint.examples, 1):
        lines.extend([f"### {number}. {row.context}, target {row.target_index}", "",
                      f"u={row.u:.12g}; d={row.d:g}; k={row.k}; p_hat={row.p_hat:g}; "
                      f"weight={row.weight:.12g}; a_c={row.a_c:g}.", "", "Source:", "",
                      "```text", *row.state, "```", "", "Target:", "", "```text", *row.subgoal, "```", ""])
    return "\n".join(lines)


def export_bundle(calibration_path: Path, output: Path) -> GeneratorCheckpoint:
    """Validate raw evidence, publish immutable artifacts, and verify the result."""
    calibration_path, output = calibration_path.resolve(), output.resolve()
    calibration: JSONObject = json.loads(calibration_path.read_text())
    hashes = source_hashes(calibration_path, calibration)
    config = configurations()["planned_board_low"]
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = export_generator_checkpoint(
        output, config=config, policy_fingerprint=calibration["policy_fingerprint"],
        calibration=calibration, source_hashes=hashes,
    )
    checkpoint = load_generator_checkpoint(output, model=config.model,
        policy_fingerprint=config.locked_policy_fingerprint, action_cap=config.calibration.action_cap,
        provider="openai", verify_sources=True)
    inference = config.model_copy(update={"generator_checkpoint": str(output),
                                          "generator_example_limit": checkpoint.actual_example_count})
    write_once(output.parent / "config.json", inference.model_dump(mode="json"))
    write_once(output.parent / "generator_inputs.json", prompt_bundle(inference, checkpoint))
    readme = output.parent / "README.md"
    content = inspection_markdown(checkpoint, inference)
    if readme.exists():
        if readme.read_text() != content:
            raise ValueError("Refusing to replace a different checkpoint inspection document")
    else:
        with readme.open("x") as stream:
            stream.write(content)
    return checkpoint


def main() -> None:
    """Export only after all thirty target calibrations have completed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint = export_bundle(args.calibration, args.output)
    print(f"Exported {checkpoint.actual_example_count} measured examples: {args.output}")
    print(f"SHA-256 {checkpoint.sha256}; complete prompts and readable examples are adjacent.")


if __name__ == "__main__":
    main()
