"""Assess actual raw representation output without confusing missing trials with failures."""

import csv
import json
import sys

import pytest

from scripts.assess_intent_representations import assess_one
from scripts.assess_intent_representations import board_family
from scripts.assess_intent_representations import comparison_pairs
from scripts.assess_intent_representations import exposure_group
from scripts.assess_intent_representations import main
from scripts.assess_intent_representations import paired
from scripts.assess_intent_representations import plot_core
from scripts.assess_intent_representations import read_predictions
from scripts.assess_intent_representations import summarize
from scripts.assess_intent_representations import training_seed_summary
from scripts.assess_intent_representations import variant_paths

BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
GOAL = {"effects": [{"predicate": "boxes_on_goals_at_least", "count": 1}], "preserve": [], "instruction": None}
TRUE = {"effects": [{"predicate": "box_at", "row": 2, "col": 2}], "preserve": [], "instruction": None}


def record(identifier="one", layout="board"):
    return {"record_id": identifier, "layout_id": layout, "split": "test", "start_observation": BOARD,
            "evaluation_family": "real_5x5x1"}


def parser(representation, raw):
    assert representation == "json"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("bad representation") from error


def prediction(intent=GOAL, *, raw=None, identifier="one", layout="board", variant="json_zero"):
    return {"record_id": identifier, "layout_id": layout, "variant": variant, "split": "test", "representation": "json",
            "support_set": "zero", "raw_output": json.dumps(intent) if raw is None else raw,
            "syntax_valid": intent is not None, "parsed_intent": intent,
            "generation_seconds": None, "generated_tokens": None}


def run(row, pred, *, bfs=False):
    return assess_one(row, pred, parser, bfs=bfs, max_depth=4, max_nodes=100, seconds=1)


def test_parser_format_semantics_and_whole_goal_are_distinct():
    good = run(record(), prediction())
    already_true = run(record(), prediction(TRUE))
    bad_format = run(record(), prediction(None, raw="```json\n{}\n```"))
    assert good["syntax_valid"] and good["semantic_valid"] and good["whole_goal_only"]
    assert not good["valid_not_whole_goal"]
    assert already_true["syntax_valid"] and not already_true["semantic_valid"]
    assert "already true" in " ".join(already_true["semantic_errors"])
    assert not bad_format["syntax_valid"] and not bad_format["semantic_valid"]
    assert bad_format["fence_removed"] and bad_format["fence_syntax_valid"]


def test_posthoc_fence_metrics_partition_valid_intents_on_the_current_board():
    not_whole = {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": [], "instruction": None}

    def fenced(intent):
        return prediction(None, raw=f"```json\n{json.dumps(intent)}\n```")

    whole = run(record("one", "board1"), fenced(GOAL))
    subgoal = run(record("two", "board2"), fenced(not_whole))
    already_true = run(record("three", "board3"), fenced(TRUE))
    for row in (whole, subgoal, already_true):
        assert not row["semantic_valid"] and row["fence_removed"]
        assert row["fence_whole_goal"] + row["fence_valid_not_whole_goal"] == row["fence_semantic_valid"]
    assert whole["fence_whole_goal"] and not whole["fence_valid_not_whole_goal"]
    assert subgoal["fence_valid_not_whole_goal"] and not subgoal["fence_whole_goal"]
    assert not already_true["fence_semantic_valid"]
    conjunction = {**GOAL, "effects": [*GOAL["effects"], *not_whole["effects"]]}
    assert run(record(), fenced(conjunction))["fence_whole_goal"]

    missing = run(record("four", "board4"), None)
    error = run(record("five", "board5"), {"status": "infrastructure_error", "error": "timeout"})
    summary = summarize("model", [whole, subgoal, already_true, missing, error], bfs=False)
    assert (summary["planned"], summary["outputs"], summary["missing"], summary["infrastructure_errors"]) == (5, 3, 1, 1)
    assert (summary["fence_semantic_valid"], summary["fence_whole_goal"], summary["fence_valid_not_whole_goal"]) == (2, 1, 1)
    assert summary["fence_whole_goal_board_mean"] == pytest.approx(1 / 3)
    assert summary["fence_valid_not_whole_goal_board_mean"] == pytest.approx(1 / 3)
    assert summary["semantic_valid"] == 0


def test_posthoc_fence_pairing_uses_identical_available_contexts():
    not_whole = {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": [], "instruction": None}

    def fenced(intent):
        return prediction(None, raw=f"```json\n{json.dumps(intent)}\n```")

    rows = []
    for identifier, before, after in (("a", not_whole, GOAL), ("b", GOAL, GOAL), ("c", None, not_whole)):
        context = record(identifier, identifier)
        for variant, intent in (("base", before), ("new", after)):
            rows.append({"variant": variant, **run(context, fenced(intent) if intent is not None else None)})
    whole = paired(rows, "base", "new", "fence_whole_goal")
    subgoal = paired(rows, "base", "new", "fence_valid_not_whole_goal")
    assert whole["planned_pairs"] == subgoal["planned_pairs"] == 3
    assert whole["paired_contexts"] == subgoal["paired_contexts"] == 2
    assert whole["unpaired_contexts"] == subgoal["unpaired_contexts"] == 1
    assert whole["board_mean_delta"] == .5 and subgoal["board_mean_delta"] == -.5


def test_json_grammar_validity_is_separate_from_schema():
    def require_intent(representation, raw):
        value = json.loads(raw)
        if not isinstance(value, dict) or not value.get("effects"):
            raise ValueError("missing effects")
        return value
    pred = prediction(None, raw="{}")
    row = assess_one(record(), pred, require_intent, bfs=False, max_depth=4, max_nodes=100, seconds=1)
    assert row["raw_json_valid"] and not row["syntax_valid"] and not row["semantic_valid"]


def test_raw_answer_is_reparsed_and_disagreement_fails():
    fraudulent = prediction()
    fraudulent["raw_output"] = "not JSON"
    with pytest.raises(ValueError, match="re-parsing"):
        run(record(), fraudulent)


def test_bounded_search_uses_real_environment_and_is_not_policy_success():
    measured = run(record(), prediction(), bfs=True)
    assert measured["solver"]["witness_found"]
    assert measured["solver"]["witness_actions"] == [3]
    assert "controller_success" not in measured


def test_missing_and_infrastructure_failures_are_not_model_failures():
    good = run(record(), prediction())
    missing = run(record("two", "second"), None)
    interrupted = run(record("three", "third"), {"status": "infrastructure_error", "error": "timeout"})
    summary = summarize("json_zero", [good, missing, interrupted], bfs=False)
    assert (summary["planned"], summary["outputs"], summary["missing"], summary["infrastructure_errors"]) == (3, 1, 1, 1)
    assert summary["semantic_valid"] == 1 and summary["semantic_valid_board_mean"] == 1
    assert summary["semantic_valid_95_board_bootstrap"] is None and summary["bfs_found"] is None
    assert summary["generation_seconds"] is None and summary["generated_tokens"] is None


def test_pairing_uses_boards_as_units_and_excludes_missing_contexts():
    rows = []
    for identifier, layout, base, candidate in [("a", "board1", False, True), ("b", "board1", False, True),
                                                 ("c", "board2", True, False)]:
        rows.extend([{"variant": "base", "record_id": identifier, "layout_id": layout, "family": "x4",
                      "status": "evaluated", "semantic_valid": base},
                     {"variant": "new", "record_id": identifier, "layout_id": layout, "family": "x4",
                      "status": "evaluated", "semantic_valid": candidate}])
    rows.extend([{"variant": "base", "record_id": "d", "layout_id": "board3", "family": "x4",
                  "status": "evaluated", "semantic_valid": True},
                 {"variant": "new", "record_id": "d", "layout_id": "board3", "family": "x4", "status": "missing"}])
    result = paired(rows, "base", "new", "semantic_valid")
    assert result["planned_pairs"] == 4 and result["paired_contexts"] == 3 and result["unpaired_contexts"] == 1
    assert result["paired_boards"] == 2 and result["board_mean_delta"] == 0 and result["bootstrap_low"] is None


def test_paired_exposure_filter_uses_only_the_requested_population():
    rows = []
    for identifier, exposure, before, after in (("a", "in_previous_lora_matrix", False, True),
                                                 ("b", "not_in_previous_lora_matrix", True, False)):
        for variant, value in (("base", before), ("new", after)):
            rows.append({"variant": variant, "record_id": identifier, "layout_id": identifier, "family": "x4",
                         "previous_lora_exposure": exposure, "status": "evaluated", "semantic_valid": value})
    previous = paired(rows, "base", "new", "semantic_valid", exposure="in_previous_lora_matrix")
    newer = paired(rows, "base", "new", "semantic_valid", exposure="not_in_previous_lora_matrix")
    assert previous["paired_contexts"] == newer["paired_contexts"] == 1
    assert previous["board_mean_delta"] == 1 and newer["board_mean_delta"] == -1


def test_frozen_prediction_rejects_foreign_contexts_and_wrong_data_hash(tmp_path):
    data, ids, path = (tmp_path / name for name in ("data.jsonl", "ids.json", "predictions.jsonl"))
    data.write_text("{}\n", encoding="utf-8")
    ids.write_text("{}\n", encoding="utf-8")
    path.write_text(json.dumps(prediction()) + "\n", encoding="utf-8")
    path.with_suffix(".run.json").write_text(json.dumps({"data_sha256": "wrong", "test_ids_sha256": "wrong"}))
    with pytest.raises(ValueError, match="different dataset"):
        read_predictions(path, "json_zero", [record()], data, ids)
    path.with_suffix(".run.json").unlink()
    with pytest.raises(ValueError, match="no run metadata"):
        read_predictions(path, "json_zero", [record()], data, ids)
    with pytest.raises(ValueError, match="frozen context"):
        read_predictions(path, "json_other", [record()], data, ids)


def test_interrupted_tail_is_missing_not_a_bad_answer(tmp_path):
    data, ids, path = (tmp_path / name for name in ("data.jsonl", "ids.json", "predictions.jsonl"))
    data.write_text("{}\n", encoding="utf-8")
    ids.write_text("{}\n", encoding="utf-8")
    raw = json.dumps(prediction()) + '\n{"record_id":'
    path.write_text(raw, encoding="utf-8")
    import hashlib
    path.with_suffix(".run.json").write_text(json.dumps({"variant": "json_zero",
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "test_ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest()}))
    rows, info = read_predictions(path, "json_zero", [record()], data, ids)
    assert list(rows) == ["one"] and info["partial_final_line"] and path.read_text() == raw


def test_family_fallback_and_duplicate_variant_rejected():
    unknown = record()
    del unknown["evaluation_family"]
    assert board_family(unknown) == "5x5x1"
    assert exposure_group(unknown) == "unknown"
    assert exposure_group({"previously_evaluated_in_lora_matrix": False}) == "not_in_previous_lora_matrix"
    assert exposure_group({"previously_evaluated_in_lora_matrix": True}) == "in_previous_lora_matrix"
    with pytest.raises(ValueError, match="distinct"):
        variant_paths(["x=a", "x=b"])


def test_explicit_contrasts_are_validated_without_reading_results():
    assert comparison_pairs(["zero", "json", "formal"], "zero", ["json=formal"]) == [
        ("zero", "json"), ("zero", "formal"), ("json", "formal")]
    with pytest.raises(ValueError, match="distinct listed"):
        comparison_pairs(["zero", "json"], "zero", ["json=other"])


def test_training_seed_coverage_keeps_partial_and_missing_out_of_descriptive_range():
    rows = [{"variant": "rep_lora_json_zero", "planned": 10, "outputs": 10, "boards_observed": 10,
             "semantic_valid_board_mean": .6, "valid_not_whole_goal_board_mean": .5},
            {"variant": "rep_lora_json_seed2718", "planned": 10, "outputs": 5, "boards_observed": 5,
             "semantic_valid_board_mean": .9, "valid_not_whole_goal_board_mean": .9}]
    points, stats = training_seed_summary(rows)
    assert [r["status"] for r in points[:3]] == ["complete", "partial", "missing"]
    result = next(r for r in stats if r["representation"] == "json" and r["metric"] == "semantic_valid")
    assert result["seeds_complete"] == 1 and result["mean_over_complete_seeds"] == result["max_over_complete_seeds"] == .6


def test_core_figure_shows_predefined_methods_even_when_others_are_missing(tmp_path):
    row = summarize("base_json_zero", [run(record(), prediction())], bfs=False)
    plot_core(tmp_path, [row])
    assert (tmp_path / "core_representations.pdf").exists()


def test_end_to_end_creates_data_and_figures_for_complete_and_missing_rows(tmp_path, monkeypatch):
    import hashlib

    data, ids, predictions = (tmp_path / name for name in ("data.jsonl", "ids.json", "predictions.jsonl"))
    first, second = record(), record("two", "board2")
    first["previously_evaluated_in_lora_matrix"] = True
    second["previously_evaluated_in_lora_matrix"] = False
    data.write_text("\n".join(map(json.dumps, [first, second])) + "\n", encoding="utf-8")
    ids.write_text(json.dumps({"record_ids": ["one", "two"],
                               "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}))
    predictions.write_text(json.dumps(prediction()) + "\n", encoding="utf-8")
    predictions2 = tmp_path / "predictions2.jsonl"
    predictions2.write_text("\n".join(map(json.dumps, [prediction(TRUE, variant="json_other"),
                                                       prediction(identifier="two", layout="board2", variant="json_other")])) + "\n")
    bad = tmp_path / "missing_run.jsonl"
    bad.write_text(json.dumps(prediction(variant="json_bad")) + "\n")
    for variant, path in (("json_zero", predictions), ("json_other", predictions2)):
        path.with_suffix(".run.json").write_text(json.dumps({"variant": variant,
            "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "test_ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest()}))
    contrasts_plan = tmp_path / "contrasts_plan.json"
    contrasts_plan.write_text(json.dumps({"status": "frozen_before_test_predictions", "primary_metric": "semantic_valid",
                                           "primary_contrasts": [["json_zero", "json_other"]], "secondary_contrasts": []}))
    output = tmp_path / "assessment"
    monkeypatch.setattr(sys, "argv", ["assessment", "--data", str(data), "--test-ids", str(ids),
                                      "--output", str(output), "--variant", f"json_zero={predictions}",
                                      "--variant", f"json_other={predictions2}", "--variant", f"json_bad={bad}",
                                      "--reference", "json_zero", "--contrasts-plan", str(contrasts_plan),
                                      "--contrast", "json_zero=json_other", "--bfs"])
    main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["results"][0]["planned"] == 2 and summary["results"][0]["missing"] == 1
    assert summary["results"][0]["bfs_found"] == 1
    assert summary["results"][0]["fence_whole_goal"] == 1
    assert summary["results"][0]["fence_valid_not_whole_goal"] == 0
    rejected = next(row for row in summary["results"] if row["variant"] == "json_bad")
    assert rejected["outputs"] == 0 and rejected["infrastructure_errors"] == 2
    assert json.loads((output / "rejected_variants.json").read_text())[0]["variant"] == "json_bad"
    assert {row["previous_lora_exposure"] for row in summary["exposure_results"]} == {
        "in_previous_lora_matrix", "not_in_previous_lora_matrix"}
    cohorts = {row["previous_lora_exposure"]: row for row in summary["paired"]
               if row["metric"] == "semantic_valid" and row["family"] == "all" and row["candidate"] == "json_other"}
    assert cohorts["in_previous_lora_matrix"]["paired_contexts"] == 1
    assert cohorts["not_in_previous_lora_matrix"]["paired_contexts"] == 0
    assert cohorts["in_previous_lora_matrix"]["contrast_role"] == "primary"
    assert cohorts["in_previous_lora_matrix"]["registered_primary_metric"]
    posthoc = [row for row in summary["paired"] if row["metric"] in
               {"fence_whole_goal", "fence_valid_not_whole_goal"}]
    assert posthoc and all(row["metric_protocol"] == "posthoc_fence" and not row["registered_primary_metric"]
                           for row in posthoc)
    assert (output / "summary.csv").exists() and (output / "exposure.csv").exists()
    for filename in ("summary.csv", "exposure.csv"):
        with (output / filename).open(newline="", encoding="utf-8") as stream:
            fields = csv.DictReader(stream).fieldnames
        assert fields is not None and {"fence_whole_goal", "fence_valid_not_whole_goal"} <= set(fields)
    with (output / "paired.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert {row["metric"] for row in csv_rows} >= {"fence_whole_goal", "fence_valid_not_whole_goal"}
    assert (output / "validity_Wszystkie_plansze.pdf").exists()
    snapshot = output / "input_snapshot"
    assert (snapshot / "json_zero.jsonl").read_bytes() == predictions.read_bytes()
    assert (snapshot / "json_zero.run.json").read_bytes() == predictions.with_suffix(".run.json").read_bytes()

    # A later append to the live generator output must not change the
    # reproducible partial analysis that used the earlier exact bytes.
    with predictions.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(prediction(identifier="two", layout="board2")) + "\n")
    reproduced = tmp_path / "reproduced"
    command = list(sys.argv)
    command[command.index("--output") + 1] = str(reproduced)
    command.extend(["--input-snapshot", str(snapshot)])
    monkeypatch.setattr(sys, "argv", command)
    main()
    repeated = json.loads((reproduced / "summary.json").read_text())
    assert repeated["results"][0]["outputs"] == 1 and repeated["results"][0]["missing"] == 1
