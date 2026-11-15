"""The final report is offline and remains useful for interrupted stages."""

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import nbformat

from scripts.build_luna_policy_followup_report import write_report


def rendered_text(path: Path) -> str:
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)
    cells = [cell for cell in document.cells if cell.cell_type == "code"]
    assert cells and all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    return "\n".join(output.text for cell in cells for output in cell.outputs if output.output_type == "stream")


def test_report_with_missing_inputs_does_not_invent_results(tmp_path):
    output = tmp_path / "report.ipynb"
    write_report(tmp_path / "not-started", output, root=tmp_path)
    text = rendered_text(output)
    assert "Brak manifestu supervisora" in text
    assert "Brak zapisanych wyników kalibracji" in text
    assert "Brak zapisanych wyników generatorów" in text
    assert "Brak CSV porównań D–N, N–U i T–U" in text
    assert "Brak zweryfikowanej komendy wznowienia" in text


def test_report_separates_infrastructure_unknown_and_saved_actions(tmp_path):
    run = tmp_path / "run"
    (run / "budget").mkdir(parents=True)
    (run / "pilot").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"status": "deadline_reached", "clock": {}, "stages": []}))
    (run / "pilot/status.json").write_text(json.dumps({"state": "stopped_api_error", "planned_attempts": 2}))
    (run / "pilot/calibrated.jsonl").write_text(json.dumps({
        "record_id": "case1", "context_id": "context1",
        "policy_measurements": {"complete": False, "status": "infrastructure_error", "attempts": [],
                                "success_fraction": None, "mean_success_actions": None, "u": None,
                                "weight": None}
    }) + "\n")
    (run / "pilot/results.jsonl").write_text(json.dumps({
        "result": {"status": "api_error", "actions": [0, 1], "model_calls": 3,
                   "steps": [{"no_op": True}, {"no_op": False}]}
    }) + "\n" + '{"interrupted":')
    with sqlite3.connect(run / "budget/luna_intent_ledger.sqlite3") as database:
        database.execute("CREATE TABLE attempts (status TEXT, reserved_nanos INTEGER, charged_nanos INTEGER)")
        database.executemany("INSERT INTO attempts VALUES (?, ?, ?)", [
            ("settled", 600000000, 1000000), ("unknown", 600000000, None)
        ])
    (run / "continuation_commands.txt").write_text("Inspect the retained unknown reservation first.\n")
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "deadline_reached" in text and "stopped_api_error" in text
    assert "Pominięto niedomknięty ostatni wiersz" in text
    assert "zapisane akcje: 2 no-op: 1 wywołania modelu: 3" in text
    assert "unknown liczba= 1 nierozliczone rezerwacje USD= 0.6" in text
    assert "infrastructure_error | [] | None | None | None | None" in text
    assert "Inspect the retained unknown reservation first." in text


def test_report_embeds_distinct_figures_for_all_completed_cohorts(tmp_path):
    run = tmp_path / "run"
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    for name, boards in (("assessment", 24), ("assessment_historical_x10", 20),
                         ("assessment_100", 100), ("assessment_historical_x10_50", 50)):
        assessment = run / name
        assessment.mkdir(parents=True)
        methods = [{"method": arm, "boards": boards, "requested_outputs": boards * 2,
                    "strict_valid_json": 8, "valid_json": 10, "valid_intents": 10, "paths_found": 7,
                    "no_path_found_among_valid": 3, "infrastructure_missing": 0,
                    "solver_infrastructure_errors": 0} for arm in ("B", "D", "R")]
        (assessment / "summary.json").write_text(json.dumps({"methods": methods, "solver": {}, "paired": {}}))
        for required in ("summary.csv", "results.jsonl", "examples.md", "validity_bounded_search.pdf",
                         "validity_bounded_search.svg"):
            (assessment / required).touch()
        (assessment / "validity_bounded_search.png").write_bytes(pixel)
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    document = nbformat.read(output, as_version=4)
    figures = [output for cell in document.cells if cell.cell_type == "code" for output in cell.outputs
               if output.output_type == "display_data" and "image/png" in output.data]
    assert len(figures) == 4
    text = rendered_text(output)
    assert "realne 12×12×4 | planowane plansze: 24" in text
    assert "historyczne, syntetyczne 12×12×10 | planowane plansze: 20" in text
    assert "rozszerzone realne 12×12×4 (24 + 76) | planowane plansze: 100" in text
    assert "rozszerzone historyczne 12×12×10 (20 + 30) | planowane plansze: 50" in text
    assert "Poprawne zamiary: 10 / 48" in text and "Poprawne zamiary: 10 / 40" in text
    assert "BFS znalazł sekwencję: 7 / 48" in text and "BFS znalazł sekwencję: 7 / 40" in text
    assert "Poprawne zamiary: 10 / 200" in text and "Poprawne zamiary: 10 / 100" in text
    prose = "\n".join(cell.source for cell in document.cells if cell.cell_type == "markdown")
    assert "Po obejrzeniu wstępnych wyników" in prose and "eksploracyjny" in prose
    assert "`core_pairs`" in prose and "nie są od nich niezależne" in prose


def test_report_does_not_open_an_unfinished_local_assessment(tmp_path):
    run = tmp_path / "run"
    assessment = run / "assessment"
    assessment.mkdir(parents=True)
    (assessment / "summary.json").write_text("THIS IS NOT FINISHED JSON")
    permutation = run / "permutation"
    permutation.mkdir()
    (permutation / "status.json").write_text('{"state":"running"}')
    (permutation / "summary.json").write_text("THIS IS NOT FINISHED JSON")
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "Brak ukończonej oceny lokalnej" in text
    assert "Brak ukończonego testu permutacyjnego" in text
    assert "Nie można odczytać" not in text


def test_replay_audit_is_displayed_only_when_source_hashes_match(tmp_path):
    run = tmp_path / "run"
    assessment = run / "assessment_100"
    assessment.mkdir(parents=True)
    results = assessment / "results.jsonl"
    summary = assessment / "summary.json"
    predictions = run / "predictions.jsonl"
    for path in (results, summary, predictions):
        path.write_text("{}\n")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {"results_sha256": digest(results), "summary_sha256": digest(summary),
             "prediction_sources": [{"path": str(predictions), "sha256": digest(predictions)}],
             "prompt_matches": 2, "prompt_pairs": 2, "replayed_paths": 2, "replayed_actions": 5}
    (run / "independent_audit_x4_100.json").write_text(json.dumps(audit))
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "zgodne pary promptów B/D: 2 / 2 | odtworzone ścieżki: 2 | akcje: 5" in text
    predictions.write_text("changed\n")
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "źródła nie zgadzają się z hashami audytu" in text
    assert "odtworzone ścieżki: 2 | akcje: 5" not in text


def test_completed_permutation_shows_distribution_not_confidence_interval(tmp_path):
    run = tmp_path / "run"
    permutation = run / "permutation"
    permutation.mkdir(parents=True)
    (permutation / "status.json").write_text('{"state":"complete"}')
    (permutation / "protocol.json").write_text(json.dumps({"permutations": 3, "seed": 5, "slot": .25}))
    method = {"family": "f", "arm": "D", "requested": 2, "observed_valid": 2,
              "permuted_valid_fraction_requested_mean": .5,
              "permuted_valid_fraction_requested_quantiles": {"q025": 0, "median": .5, "q975": 1},
              "permutation_fraction_at_least_observed": .1, "generator_infrastructure_missing": 0,
              "original_solver": {"paths_found": 2, "requested": 2},
              "first_permutation_solver": {"paths_found": 1, "requested": 2}}
    figures = ["permutation_distribution_f.png", "permutation_distribution_f.pdf", "permutation_distribution_f.svg"]
    (permutation / "summary.json").write_text(json.dumps({"methods": [method], "figures": figures}))
    (permutation / "permutation_distribution.csv").write_text(
        "family,arm,permutation_index\nf,D,0\nf,D,1\nf,D,2\n")
    for name in ("permutations.json", "paired_cases.jsonl", figures[1], figures[2]):
        (permutation / name).touch()
    (permutation / figures[0]).write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "Liczba losowych przypisań: 3 | seed: 5 | wybrany slot: 0.25" in text
    assert "poprawne na oryginalnej planszy: 2 / 2" in text
    assert "oryginał: 2 / 2 | pierwsze przypisanie: 1 / 2" in text
    prose = "\n".join(cell.source for cell in nbformat.read(output, as_version=4).cells
                      if cell.cell_type == "markdown")
    assert "nie są przedziałem ufności" in prose


def test_local_diagnostics_use_board_denominators_and_show_illustrative_examples(tmp_path):
    run = tmp_path / "run"
    assessment = run / "assessment"
    assessment.mkdir(parents=True)
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    methods = [{"method": arm, "boards": 2, "requested_outputs": 4} for arm in ("B", "D", "R")]
    (assessment / "summary.json").write_text(json.dumps({"methods": methods, "solver": {}, "paired": {}}))
    for name in ("summary.csv", "examples.md", "validity_bounded_search.pdf", "validity_bounded_search.svg"):
        (assessment / name).touch()
    (assessment / "validity_bounded_search.png").write_bytes(pixel)
    repeated_intent = {"effects": [{"predicate": "box_at", "row": 4, "col": 1}], "preserve": []}
    goal_intent = {"effects": [{"predicate": "boxes_on_goals_at_least", "count": 10}], "preserve": []}
    diagnostic = {"methods": {arm: {"boards": 2, "outputs": 4, "valid_boards": 1,
                                      "path_found_boards": 1, "slots_with_identical_raw_output": 2,
                                      "validation_errors": {"position is a wall": 2}}
                              for arm in ("B", "D", "R")}}
    diagnostic["methods"]["D"]["most_common_parseable_intents"] = [{"intent": repeated_intent, "count": 4}]
    (assessment / "generation_diagnostics.json").write_text(json.dumps(diagnostic))
    rows = []
    for board in ("a", "b"):
        for arm in ("B", "D", "R"):
            for _ in range(2):
                valid = arm == "R" or (arm == "D" and board == "a")
                row = {"layout_id": board, "arm": arm, "start_observation": ["#@$$$$#"],
                       "parsed": goal_intent if arm == "B" else repeated_intent, "valid_intent": valid,
                       "prediction_status": "valid_intent" if valid else "invalid_intent",
                       "validation_errors": [] if valid else ["position is a wall"], "path_found": valid,
                       "goal_box_change": int(arm == "R" and board == "a"),
                       "search": {"status": "witness_found", "witness_found": True, "witness_actions": [3],
                                  "reached_observation": ["#-@$$$#"]} if valid else None}
                rows.append(row)
    (assessment / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "Żądane liczby skrzynek na celach — wyjścia: {10: 4} | plansze: {10: 2}" in text
    assert "poprawny na planszach: 1 / 2" in text
    assert "Plansze z większą liczbą skrzynek na celach po ścieżce BFS: 1 / 2" in text
    assert "D: poprawny zamiar i sekwencja BFS" in text and "Plansza po sekwencji BFS:" in text
    assert "BFS: znaleziono sekwencję | akcje: [3]" in text
    assert "D: niepoprawny zamiar" in text and "position is a wall" in text
    assert "BFS nie był uruchomiony; brak stanu po sekwencji." in text


def test_historical_decrease_is_shown_as_a_local_bfs_outcome(tmp_path):
    run = tmp_path / "run"
    assessment = run / "assessment_historical_x10"
    assessment.mkdir(parents=True)
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    methods = [{"method": arm, "boards": 1, "requested_outputs": 2,
                "paths_increasing_boxes_on_goals": 0, "paths_decreasing_boxes_on_goals": 2 if arm == "D" else 0}
               for arm in ("B", "D", "R")]
    (assessment / "summary.json").write_text(json.dumps({"methods": methods, "solver": {}, "paired": {}}))
    for name in ("summary.csv", "examples.md", "validity_bounded_search.pdf", "validity_bounded_search.svg"):
        (assessment / name).touch()
    (assessment / "validity_bounded_search.png").write_bytes(pixel)
    diagnostic = {"methods": {arm: {"boards": 1, "outputs": 2, "valid_boards": 1,
                                      "path_found_boards": 1, "slots_with_identical_raw_output": 1}
                              for arm in ("B", "D", "R")}}
    (assessment / "generation_diagnostics.json").write_text(json.dumps(diagnostic))
    row = {"layout_id": "historical", "arm": "D", "start_observation": ["#@**#"],
           "parsed": {"effects": [{"predicate": "box_at", "row": 1, "col": 1}], "preserve": []},
           "valid_intent": True, "path_found": True, "goal_box_change": -2,
           "search": {"witness_actions": [1], "reached_observation": ["#@$$#"]}}
    (assessment / "results.jsonl").write_text((json.dumps(row) + "\n") * 2)
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "(paths_decreasing_boxes_on_goals): 2" in text
    assert "| z mniejszą: 1 / 1" in text and "{-2: 1}" in text
    assert "Lokalny BFS znalazł sekwencję: [1]" in text
    assert "Zmiana liczby skrzynek na celach: -2" in text
    prose = "\n".join(cell.source for cell in nbformat.read(output, as_version=4).cells
                      if cell.cell_type == "markdown")
    assert "To nie jest próba Luny" in prose


def test_old_results_keep_board_and_fragment_denominators(tmp_path):
    old = tmp_path / "analysis/intent_generator_20261114T032232Z"
    (old / "fence_feasibility").mkdir(parents=True)
    (old / "figures").mkdir()
    (old / "figures/posthoc_feasibility.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    (old / "dataset_manifest.json").write_text(json.dumps({"counts": {
        "train": {"boards": 9, "records": 358}, "validation": {"boards": 3, "records": 66},
        "test": {"boards": 20, "records": 100}
    }}))
    (old / "fence_feasibility/summary.csv").write_text(
        "method,outputs,valid_nontrivial,witness_found,unresolved_among_valid\n"
        "base,100,66,7,59\nadapted,100,66,44,22\n"
    )
    run = tmp_path / "run"
    adapter = run / "adapters/D_retry"
    adapter.mkdir(parents=True)
    (adapter / "training_curve.jsonl").write_text(
        json.dumps({"step": 1, "train_loss": 1.0}) + "\n"
        + json.dumps({"step": 2, "train_loss": .5, "validation_loss": .7}) + "\n"
    )
    output = tmp_path / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "train independent boards: 9 checked fragments: 358" in text
    assert "test independent boards: 20 checked fragments: 100" in text
    assert "base valid non-trivial: 66/100 BFS sequence found: 7/100 valid but unresolved: 59/66" in text
    assert "adapted valid non-trivial: 66/100 BFS sequence found: 44/100 valid but unresolved: 22/66" in text
    assert "Previous exploratory study — post-hoc fence normalisation" in text
    assert "zapisane kroki: 2 | walidacje: 1" in text
    assert "nie jest głównym wynikiem porównania" in text
    document = nbformat.read(output, as_version=4)
    figures = [item for cell in document.cells if cell.cell_type == "code" for item in cell.outputs
               if item.output_type == "display_data" and "image/png" in item.data]
    assert len(figures) == 2


def test_report_displays_saved_paired_prompts_and_controller_request_without_credentials(tmp_path):
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "predictions").mkdir()
    (run / "pilot_continued").mkdir()
    (run / "budget_carried/luna_requests").mkdir(parents=True)
    (run / "inputs/test_first_block.jsonl").write_text(json.dumps({"record_id": "fixed-board"}) + "\n")
    for arm in ("B", "D"):
        row = {"record_id": "fixed-board", "arm": arm, "status": "valid",
               "prompt": [{"role": "system", "content": f"{arm} FULL SYSTEM\nline two"},
                          {"role": "user", "content": "FULL BOARD\n#@$.#"}],
               "raw_output": f"FULL {arm} OUTPUT"}
        (run / f"predictions/{arm}_slot0.jsonl").write_text(json.dumps(row) + "\n")
        metadata = {"status": "complete", "model_weights_sha256": "weights", "model_config_sha256": "config",
                    "tokenizer_sha256": "tokenizer", "code_sha256": "code", "dtype": "float16",
                    "decoding": "greedy", "max_length": 512, "max_new_tokens": 128,
                    "enable_thinking": False, "revision": "revision", "seed": 1}
        (run / f"predictions/{arm}_slot0.run.json").write_text(json.dumps(metadata))
    controller_prompt = "FULL LUNA PROMPT\ncurrent board and attempt history"
    result = {"record_id": "controller-case", "attempt": 1,
              "result": {"status": "invalid_response", "actions": [], "model_calls": 1,
                         "steps": [{"index": 0, "prompt": controller_prompt}]}}
    (run / "pilot_continued/results.jsonl").write_text(json.dumps(result) + "\n")
    audit = {"tag": "final:controller-case:attempt:1", "status": "settled",
             "request": {"input": controller_prompt, "api_key": "PRIVATE-DO-NOT-DISPLAY"}}
    (run / "budget_carried/luna_requests/request.json").write_text(json.dumps(audit))
    output = run / "report.ipynb"
    write_report(run, output, root=tmp_path)
    text = rendered_text(output)
    assert "Pierwszy zamrożony kontekst: fixed-board | slot: 0" in text
    assert "B FULL SYSTEM\nline two" in text and "D FULL SYSTEM\nline two" in text
    assert "FULL BOARD\n#@$.#" in text
    assert "FULL B OUTPUT" in text and "FULL D OUTPUT" in text
    assert controller_prompt in text and "dokładnych dopasowań w audycie: 1" in text
    assert "identyczne zapisane wiadomości B/D: 0 / 1" in text
    assert "rozbieżne lub brakujące pola porównania: []" in text
    assert "PRIVATE-DO-NOT-DISPLAY" not in text
