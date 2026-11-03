"""Order measured demonstrations to give the small prefix several time levels."""

import json
from pathlib import Path

from carl.icl.checkpoints import export_generator_checkpoint, load_generator_checkpoint
from carl.icl.types import JSONObject
from scripts.generator_probe.export_checkpoint import (
    DEFAULT_CALIBRATION, file_digest, inspection_markdown, prompt_bundle, source_hashes,
)
from scripts.generator_probe.prepare_dataset import write_once
from scripts.luna_generator_probe import OUTPUT, configurations


def ordered_ids(calibration: JSONObject) -> list[tuple[str, int]]:
    """Rotate within-context time order across contexts, without validation results."""
    groups: dict[str, list[JSONObject]] = {}
    for row in calibration["examples"]:
        groups.setdefault(row["context"], []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["d"], row["k"], row["target_index"]))
    selected: list[tuple[str, int]] = []
    for cycle in range(max(map(len, groups.values()))):
        for context_index, group in enumerate(groups.values()):
            row = group[(context_index + cycle) % len(group)]
            key = (row["context"], row["target_index"])
            if key not in selected:
                selected.append(key)
    return selected


def main() -> None:
    calibration = json.loads(DEFAULT_CALIBRATION.read_text())
    directory = OUTPUT / "checkpoint_balanced"
    directory.mkdir(parents=True, exist_ok=True)
    selection = directory / "selection.json"
    ids = ordered_ids(calibration)
    write_once(selection, {"rule": "Group by context in calibration order; sort by(d,k,target_index); "
                          "cycle through contexts selecting index(context_index+cycle) modulo group size; "
                          "skip already selected IDs. No validation outcome used.",
                          "example_ids": [list(key) for key in ids]})
    hashes = source_hashes(DEFAULT_CALIBRATION, calibration)
    hashes.update({str(selection.resolve()): file_digest(selection),
                   str(Path(__file__).resolve()): file_digest(Path(__file__))})
    config = configurations()["planned_board_low"]
    output = directory / "generator.json"
    checkpoint = export_generator_checkpoint(
        output, config=config, policy_fingerprint=calibration["policy_fingerprint"],
        calibration=calibration, source_hashes=hashes, example_ids=ids)
    checkpoint = load_generator_checkpoint(output, verify_sources=True)
    inference = config.model_copy(update={"generator_checkpoint": str(output),
                                          "generator_example_limit": checkpoint.actual_example_count})
    write_once(directory / "config.json", inference.model_dump(mode="json"))
    write_once(directory / "generator_inputs.json", prompt_bundle(inference, checkpoint))
    content = inspection_markdown(checkpoint, inference)
    readme = directory / "README.md"
    if readme.exists() and readme.read_text() != content:
        raise ValueError("Checkpoint documentation changed")
    readme.write_text(content)
    print(f"Exported {checkpoint.actual_example_count} measured examples to {output}")
    print("First four (d,u):", [(item.d, item.u) for item in checkpoint.examples[:4]])


if __name__ == "__main__":
    main()
