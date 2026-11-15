#!/usr/bin/env python3
"""Render an explanatory HTML edition of the saved overnight campaign.

This is a presentation-only operation: it reads frozen records and never
starts an experiment.  The original compact HTML and its manifest are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/report_assets/overnight_campaign_readable.html"
DEFAULT_CAMPAIGN = ROOT / "analysis/overnight_cpu_20261115T061352Z"
PROPOSAL = (
    ROOT.parent
    / "Hierarchical_Planning_with_Policy_Relative_Temporal_Scales_15_11_2026_pre_second_review.zip"
)
LABELS = {
    "base": "Base Qwen",
    "historical_checkpoint220": "Earlier LoRA",
    "narrow": "Narrow LoRA",
    "wide": "Wide LoRA",
    "wide_aug": "Wide + D4 LoRA",
    "adasubs": "AdaSubS",
    "ksubs": "Fixed-k kSubS",
    "direct_policy": "Direct policy",
    "primitive_astar": "Primitive A*",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def code(value: str) -> str:
    return f"<pre><code>{esc(value)}</code></pre>"


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_board(
    rows: list[str], label: str, target: tuple[int, int] | None = None
) -> str:
    """Render the actual ASCII board with visible zero-based coordinates."""
    symbols = {
        "#": ("wall", ""),
        "-": ("floor", ""),
        ".": ("goal", "○"),
        "$": ("box", "■"),
        "*": ("box-goal", "◆"),
        "@": ("player", "●"),
        "+": ("player-goal", "◎"),
    }
    if len(rows) != 12 or any(len(row) != 12 for row in rows):
        raise ValueError("Expected a complete 12 by 12 board")
    result = [f'<table class="board" aria-label="{esc(label)}"><thead><tr><th></th>']
    result.extend(f'<th scope="col">{col}</th>' for col in range(12))
    result.append("</tr></thead><tbody>")
    for row_index, row in enumerate(rows):
        result.append(f'<tr><th scope="row">{row_index}</th>')
        for col_index, cell in enumerate(row):
            name, symbol = symbols[cell]
            focus = " target" if target == (row_index, col_index) else ""
            description = f"row {row_index}, column {col_index}: {name}"
            result.append(
                f'<td class="tile {name}{focus}" title="{esc(description)}">{symbol}</td>'
            )
        result.append("</tr>")
    result.append("</tbody></table>")
    return "".join(result)


def record_table(campaign: Path, board: dict[str, Any]) -> str:
    predictions = load_jsonl(campaign / "stage_04/evaluation/predictions.jsonl")
    by_method = {
        row["method"]: row
        for row in predictions
        if row["context_id"] == board["context_id"] and row["transform"] == "identity"
    }
    rows = []
    for output in board["outputs"]:
        prediction = by_method[output["method"]]
        if prediction["raw_output"] != output["raw_output"]:
            raise ValueError("Example and saved prediction disagree")
        valid = "Valid" if output["valid_nontrivial"] else "Invalid"
        errors = prediction["validation_errors"]
        reason = "; ".join(errors) if errors else "All initial checks passed"
        if output["bounded_path_found"] is True:
            search = f'Path found, {output["bfs_search"]["witness_depth"]} actions'
        elif output["bounded_path_found"] is False:
            search = f'Unresolved: {output["bfs_status"].replace("_", " ")}'
        else:
            search = "Not run for invalid intent"
        rows.append(
            f'<tr><th scope="row">{esc(LABELS[output["method"]])}</th>'
            f'<td>{code(output["raw_output"])}</td><td><strong>{valid}</strong><br>{esc(reason)}</td>'
            f"<td>{esc(search)}</td></tr>"
        )
    return "".join(rows)


def messages_html(prompt: dict[str, Any]) -> str:
    parts = []
    for message in prompt["messages"]:
        parts.append(
            f'<div class="prompt-role">{esc(message["role"])}</div>{code(message["content"])}'
        )
    return "".join(parts)


def prompt_for(
    campaign: Path, stage: str, method: str, family: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = campaign / stage
    predictions = load_jsonl(directory / "predictions.jsonl")
    prediction = next(
        row
        for row in predictions
        if row["method"] == method
        and row["family"] == family
        and row["board_rank"] == 0
        and row["transform"] == "identity"
    )
    prompt = next(
        row
        for row in load_jsonl(directory / "prompts.jsonl")
        if row["case_id"] == prediction["case_id"]
    )
    rendered_hash = hashlib.sha256(prompt["rendered_prompt"].encode()).hexdigest()
    if (
        rendered_hash != prompt["prompt_sha256"]
        or prediction["prompt_sha256"] != rendered_hash
    ):
        raise ValueError("Saved prompt hash does not match the prediction")
    return prompt, prediction


def generator_rows(campaign: Path) -> str:
    records = load_csv(campaign / "stage_04_figure_data.csv")
    output = []
    for family in ("12x12x4", "12x12x10"):
        for method in (
            "base",
            "historical_checkpoint220",
            "narrow",
            "wide",
            "wide_aug",
        ):
            row = next(
                item
                for item in records
                if item["family"] == family
                and item["method"] == method
                and item["transform"] == "identity"
                and item["metric"] == "valid"
            )
            total = row["planned_outputs"]
            output.append(
                f'<tr><td>{esc(family)}</td><th scope="row">{esc(LABELS[method])}</th>'
                f'<td>{esc(row["valid_outputs"])}/{esc(total)}</td>'
                f'<td>{esc(row["bounded_paths_found"])}/{esc(total)}</td></tr>'
            )
    return "".join(output)


def planner_rows(campaign: Path) -> str:
    records = load_csv(campaign / "stage_05/stage_05_summary.csv")
    output = []
    for family in ("12x12x4", "12x12x10"):
        for method in ("adasubs", "ksubs", "direct_policy", "primitive_astar"):
            row = next(
                item
                for item in records
                if item["family"] == family
                and item["method"] == method
                and item["source_family"] == "all"
            )
            output.append(
                f'<tr><td>{esc(family)}</td><th scope="row">{esc(LABELS[method])}</th>'
                f'<td>{esc(row["solved"])}/{esc(row["planned"])}</td>'
                f'<td>{esc(row["timeouts"])}</td><td>{esc(row["total_transitions"])}</td>'
                f'<td>{esc(row["total_generation_calls"])}</td></tr>'
            )
    return "".join(output)


def verify_manifest(campaign: Path, manifest: dict[str, Any]) -> int:
    count = 0
    for category in ("input_sha256", "output_sha256"):
        for name, expected in manifest[category].items():
            path = Path(name)
            if not path.is_absolute():
                path = campaign / path
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"{category} mismatch: {path}")
            count += 1
    return count


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def render(campaign: Path) -> tuple[str, dict[str, str]]:
    examples = load_json(campaign / "stage_04_examples.json")
    x4 = next(row for row in examples if row["family"] == "12x12x4")
    x10 = next(row for row in examples if row["family"] == "12x12x10")
    training = load_jsonl(campaign / "stage_04/data_v3/wide.jsonl")
    single = next(
        row
        for row in training
        if len(row["intent"]["effects"]) == 1
        and row["intent"]["effects"][0]["predicate"] == "box_at"
    )
    combined = next(row for row in training if row["effect_type"] == "goal_conjunction")
    for row in (single, combined):
        if not row["validation"]["accepted"]:
            raise ValueError("Selected demonstration annotation was rejected")
    prompt, prediction = prompt_for(campaign, "stage_04/evaluation", "wide", "12x12x4")
    icl_prompt, icl_prediction = prompt_for(
        campaign, "stage_03/fresh_evaluation", "in_context", "12x12x10"
    )
    if prediction["context_id"] != x4["context_id"]:
        raise ValueError("Prompt example and board example refer to different contexts")
    action_names = ("up", "down", "left", "right")
    prior = next(
        out for out in x4["outputs"] if out["method"] == "historical_checkpoint220"
    )
    search = prior["bfs_search"]
    positions = [
        (row["intent"]["effects"][0]["row"], row["intent"]["effects"][0]["col"])
        for row in (single, combined)
    ]
    substitutions = {
        "PROPOSAL_LINK": esc(os.path.relpath(PROPOSAL, campaign)),
        "SINGLE_START": render_board(
            single["start_observation"],
            "Training observation before the fragment",
            positions[0],
        ),
        "SINGLE_LATER": render_board(
            single["later_observation"], "Recorded later observation", positions[0]
        ),
        "SINGLE_INTENT": code(pretty(single["intent"])),
        "SINGLE_RECORD": esc(single["record_id"]),
        "SINGLE_ACTIONS": esc(
            " → ".join(action_names[action] for action in single["fragment_actions"])
        ),
        "SINGLE_K": str(single["k"]),
        "TRAIN_USER": code(
            "Task: "
            + single["task_goal"]
            + "\nCurrent board:\n"
            + "\n".join(single["start_observation"])
        ),
        "TRAIN_TARGET": code(
            json.dumps(single["intent"], ensure_ascii=False, separators=(",", ":"))
        ),
        "COMBINED_START": render_board(
            combined["start_observation"], "Conjunction training start", positions[1]
        ),
        "COMBINED_LATER": render_board(
            combined["later_observation"],
            "Conjunction recorded later state",
            positions[1],
        ),
        "COMBINED_INTENT": code(pretty(combined["intent"])),
        "COMBINED_RECORD": esc(combined["record_id"]),
        "COMBINED_K": str(combined["k"]),
        "X4_BOARD": render_board(
            x4["board"], "First evaluation board with four boxes", (2, 4)
        ),
        "X4_REACHED": render_board(
            search["reached_observation"], "State reached by bounded search", (2, 4)
        ),
        "X4_TABLE": record_table(campaign, x4),
        "X4_LAYOUT": esc(x4["layout_id"]),
        "X4_ACTIONS": esc(
            " → ".join(action_names[action] for action in search["witness_actions"])
        ),
        "X10_BOARD": render_board(
            x10["board"], "First evaluation board with ten boxes"
        ),
        "X10_TABLE": record_table(campaign, x10),
        "X10_LAYOUT": esc(x10["layout_id"]),
        "GENERATOR_ROWS": generator_rows(campaign),
        "PLANNER_ROWS": planner_rows(campaign),
        "PROMPT_MESSAGES": messages_html(prompt),
        "PROMPT_RENDERED": code(prompt["rendered_prompt"]),
        "PROMPT_HASH": esc(prompt["prompt_sha256"]),
        "PROMPT_OUTPUT": code(prediction["raw_output"]),
        "ICL_MESSAGES": messages_html(icl_prompt),
        "ICL_RENDERED": code(icl_prompt["rendered_prompt"]),
        "ICL_HASH": esc(icl_prompt["prompt_sha256"]),
        "ICL_OUTPUT": code(icl_prediction["raw_output"]),
    }
    result = TEMPLATE.read_text(encoding="utf-8")
    for key, value in substitutions.items():
        marker = f"@@{key}@@"
        if result.count(marker) != 1:
            raise ValueError(f"Expected exactly one {marker} in template")
        result = result.replace(marker, value)
    leftover = re.findall(r"@@[A-Z_0-9]+@@", result)
    if leftover:
        raise ValueError(f"Missing template values: {leftover}")
    files = (
        TEMPLATE,
        PROPOSAL,
        campaign / "stage_04_examples.json",
        campaign / "stage_04/data_v3/wide.jsonl",
        campaign / "stage_04/evaluation/prompts.jsonl",
        campaign / "stage_04/evaluation/predictions.jsonl",
        campaign / "stage_03/fresh_evaluation/prompts.jsonl",
        campaign / "stage_03/fresh_evaluation/predictions.jsonl",
        campaign / "stage_04_figure_data.csv",
        campaign / "stage_05/stage_05_summary.csv",
    )
    return result, {str(path): digest(path) for path in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    report = campaign / "OVERNIGHT_REPORT.html"
    original = campaign / "OVERNIGHT_REPORT_compact_original.html"
    manifest_path = campaign / "overnight_report_manifest.json"
    manifest = load_json(manifest_path)
    verified = verify_manifest(campaign, manifest)
    content, sources = render(campaign)
    old_html = report.read_bytes()
    old_manifest = manifest_path.read_bytes()
    if not original.exists():
        atomic_write(original, old_html)
        atomic_write(
            campaign / "overnight_report_manifest_compact_original.json", old_manifest
        )
    elif not (campaign / "overnight_report_manifest_compact_original.json").exists():
        raise ValueError("Original HTML exists without its original manifest")
    original_manifest = campaign / "overnight_report_manifest_compact_original.json"
    previous = load_json(original_manifest)
    original_hash = digest(original)
    if previous["output_sha256"]["OVERNIGHT_REPORT.html"] != original_hash:
        raise ValueError("Preserved original HTML does not match its original manifest")
    new_bytes = content.encode("utf-8")
    new_hash = hashlib.sha256(new_bytes).hexdigest()
    manifest["output_sha256"]["OVERNIGHT_REPORT.html"] = new_hash
    manifest["presentation_revision"] = {
        "original_html": original.name,
        "original_html_sha256": original_hash,
        "original_manifest_sha256": digest(original_manifest),
        "renderer_sha256": digest(Path(__file__)),
        "template_sha256": digest(TEMPLATE),
        "source_sha256": sources,
        "purpose": "Presentation-only HTML rewrite; saved experimental measurements are unchanged.",
    }
    audit = {
        "state": "pass",
        "checked_prior_hashes": verified,
        "html_sha256": new_hash,
        "original_html_sha256": original_hash,
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m scripts.render_overnight_readable --campaign "
        + str(campaign),
        "inputs": sources,
        "scope": "presentation only; no inference, training or API calls",
    }
    try:
        atomic_write(report, new_bytes)
        atomic_write(manifest_path, (pretty(manifest) + "\n").encode())
        verify_manifest(campaign, manifest)
        atomic_write(
            campaign / "readable_report_audit.json", (pretty(audit) + "\n").encode()
        )
    except Exception:
        atomic_write(report, old_html)
        atomic_write(manifest_path, old_manifest)
        raise
    print(
        f"Updated {report}; retained original at {original}; verified {verified} prior hashes."
    )


if __name__ == "__main__":
    main()
