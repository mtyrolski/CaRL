#!/usr/bin/env python3
"""Build and execute the offline report for the Luna Sokoban intent study."""
import argparse
import csv
import json
import os
import random
import sqlite3
import tempfile
from collections import Counter
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/luna_intent_20261114T060730Z"
COMPLETED = {"achieved", "preservation_violated", "task_solved", "action_limit", "invalid_response"}
DECISIVE = {"achieved", "preservation_violated", "task_solved", "invalid_response"}
FAMILIES = ("real_12x12x4", "proposal_gamma2_12x12x10")
PHASES = ("calibration", "test")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Retain complete lines if the writer was interrupted; report damaged lines."""
    if not path.exists():
        return [], []
    rows, errors = [], []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            rows.append(value)
        except (ValueError, TypeError) as exc:
            errors.append(f"{path.name}:{index}: {exc}")
    return rows, errors


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def board_interval(rows: list[dict[str, Any]], seed: int = 0) -> tuple[float | None, float | None]:
    """Cluster bootstrap over layouts; fewer than five layouts is descriptive."""
    boards: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        boards[row["layout_id"]].append(int(row["success"]))
    keys = sorted(boards)
    if len(keys) < 5:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(2000):
        values = [value for _ in keys for value in boards[rng.choice(keys)]]
        estimates.append(sum(values) / len(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def final_action(text: str) -> str | None:
    """Accept only an unambiguous JSON action object."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate key")
        return dict(pairs)

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return None
    if (isinstance(value, dict) and set(value) == {"action"} and isinstance(value["action"], str)
            and value["action"] in {"UP", "DOWN", "LEFT", "RIGHT"}):
        return value["action"]
    return None


def concatenation_artifacts(results: dict[str, dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Identify legacy parser failures only when the saved API trace proves them."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        tag = audit.get("tag")
        if isinstance(tag, str):
            grouped[tag.rsplit(":", 1)[-1]].append(audit)
    found = {}
    for record_id, row in results.items():
        steps = row["result"].get("steps", [])
        if row["result"].get("status") != "invalid_response" or not steps:
            continue
        raw = steps[-1].get("raw_response")
        if not isinstance(raw, str):
            continue
        matches = []
        for audit in grouped.get(record_id, []):
            response = audit.get("response")
            if not isinstance(response, dict):
                continue
            parts = [(message.get("phase"), content.get("text"), message.get("status"))
                     for message in response.get("output", []) if isinstance(message, dict)
                     and message.get("type") == "message"
                     for content in message.get("content", []) if isinstance(content, dict)
                     and content.get("type") == "output_text" and isinstance(content.get("text"), str)]
            if "".join(text for _, text, _ in parts) != raw:
                continue
            final = [(text, status) for phase, text, status in parts if phase == "final_answer"]
            action = final_action(final[0][0]) if len(final) == 1 else None
            matches.append({"reservation_id": audit.get("reservation_id"), "final_action": action,
                            "proven": len(parts) > 1 and len(final) == 1 and action is not None
                            and final[0][1] == "completed" and response.get("status") == "completed" and raw != final[0][0]})
        # If several requests yielded the same text, all must support this diagnosis.
        if matches and all(match["proven"] for match in matches):
            found[record_id] = {"record_id": record_id, "audit_matches": matches,
                                "reason": "concatenated_commentary_and_valid_final_answer"}
    return found


def audited_calibration(cases: list[dict[str, Any]], results: dict[str, dict[str, Any]],
                        artifacts: dict[str, dict[str, Any]], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Mask all u values for contexts with any interrupted or unassessed candidate."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case.get("phase") == "calibration":
            groups[case["context_id"]].append(case)
    saved = {context["context_id"]: context for context in calibration.get("contexts", [])}
    rows = []
    for context_id, candidates in groups.items():
        context = saved.get(context_id, {})
        saved_candidates = {item["record_id"]: item for item in context.get("candidates", [])}

        def assessed(case: dict[str, Any]) -> bool:
            record_id = case["record_id"]
            row = results.get(record_id)
            return bool(row and record_id not in artifacts and row["result"].get("status") in COMPLETED
                        and row["result"].get("model_calls", 0) > 0)

        valid = bool(context.get("complete") and set(saved_candidates) == {c["record_id"] for c in candidates}
                     and all(assessed(case) for case in candidates))
        for case in candidates:
            record_id = case["record_id"]
            result = results.get(record_id, {}).get("result", {})
            recorded = saved_candidates.get(record_id, {})
            rows.append({"context_id": context_id, "record_id": record_id,
                         "status": result.get("status"), "parser_artifact": record_id in artifacts,
                         "context_complete_after_audit": valid,
                         "successes": recorded.get("successes") if assessed(case) else None,
                         "success_fraction": recorded.get("success_fraction") if assessed(case) else None,
                         "mean_success_actions": recorded.get("mean_success_actions") if assessed(case) else None,
                         "u": recorded.get("u") if valid else None})
    return rows


def read_ledger(run: Path) -> dict[str, Any] | None:
    """Read the shared budget database without creating or modifying it."""
    path = run / "luna_intent_ledger.sqlite3"
    if not path.exists():
        return None
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        entries = db.execute(
            "SELECT status, COUNT(*), "
            "COALESCE(SUM(COALESCE(charged_nanos, 0)), 0), "
            "COALESCE(SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END), 0) "
            "FROM attempts GROUP BY status ORDER BY status"
        ).fetchall()
        effective = db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()
    unit = Decimal(1_000_000_000)
    costs = [{"status": status, "attempts": count, "charged_usd": str(Decimal(charged) / unit),
              "reserved_usd": str(Decimal(reserved) / unit)} for status, count, charged, reserved in entries]
    charged = sum(row[2] for row in entries)
    reserved = sum(row[3] for row in entries)
    unknown = sum(row[3] for row in entries if row[0] == "unknown")
    return {"path": str(path), "attempts": sum(row[1] for row in entries), "by_status": costs,
            "known_charge_usd": str(Decimal(charged) / unit), "retained_reservations_usd": str(Decimal(reserved) / unit),
            "unknown_reserved_usd": str(Decimal(unknown) / unit),
            "conservative_total_with_reservations_usd": str(Decimal(charged + reserved) / unit),
            "effective_limit_usd": str(Decimal(effective[0]) / unit) if effective else None}


def load_report(run: Path) -> dict[str, Any]:
    """Join frozen cases and results, never treating missing cases as failures."""
    cases, case_errors = read_jsonl(run / "data/records.jsonl")
    rollouts, rollout_errors = read_jsonl(run / "results/rollouts.jsonl")
    manifest = read_json(run / "data/manifest.json", {})
    plan = read_json(run / "experiment_plan.json", {})
    selected = set(plan.get("record_ids", [case["record_id"] for case in cases]))
    cases = [case for case in cases if case["record_id"] in selected]
    cases_by_id = {case["record_id"]: case for case in cases}
    warnings = case_errors + rollout_errors
    counts = Counter(row.get("record_id") for row in rollouts)
    duplicate_ids = {record_id for record_id, count in counts.items() if count > 1}
    if duplicate_ids:
        warnings.append(f"Duplicate rollout IDs excluded from analysis: {sorted(duplicate_ids)}")
    results = {}
    for row in rollouts:
        record_id = row.get("record_id")
        case = cases_by_id.get(record_id)
        if record_id in duplicate_ids:
            continue
        if case is None:
            warnings.append(f"Rollout not in frozen plan: {record_id}")
            continue
        if any(row.get(key) != case.get(key) for key in ("phase", "family", "layout_id", "context_id")):
            warnings.append(f"Rollout metadata differs from frozen case: {record_id}")
            continue
        result = row.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("actions"), list):
            warnings.append(f"Incomplete rollout object: {record_id}")
            continue
        results[record_id] = row
    absent_from_data = selected - set(cases_by_id)
    if absent_from_data:
        warnings.append(f"Planned IDs missing from data: {sorted(absent_from_data)}")

    audit_run = (run / plan.get("audit_run", ".")).resolve()
    audit_prefix = plan.get("audit_tag_prefix")
    all_audits, audit_errors = [], []
    for path in sorted((audit_run / "luna_requests").glob("*.json")):
        try:
            all_audits.append(read_json(path))
        except (OSError, ValueError) as exc:
            audit_errors.append(f"{path.name}: {exc}")
    warnings.extend(audit_errors)
    if audit_prefix is None:
        audits = [audit for audit in all_audits
                  if not str(audit.get("tag", "")).startswith(("structured:", "final:", "expanded:"))]
    elif isinstance(audit_prefix, str) and audit_prefix:
        audits = [audit for audit in all_audits if str(audit.get("tag", "")).startswith(audit_prefix)]
    else:
        raise ValueError("audit_tag_prefix must be a nonempty string when supplied")

    def settled_charge(rows: list[dict[str, Any]]) -> str:
        return str(sum((Decimal(str(a["conservative_charge_usd"])) for a in rows
                        if a.get("status") == "settled" and a.get("conservative_charge_usd") is not None), Decimal(0)))
    artifacts = concatenation_artifacts(results, audits)
    summaries, curve = [], []
    for phase in PHASES:
        for family in FAMILIES:
            group = [case for case in cases if case.get("phase") == phase and case.get("family") == family]
            if not group:
                continue
            observed = [results[case["record_id"]] for case in group if case["record_id"] in results]
            complete = [row for row in observed if row["result"].get("status") in COMPLETED
                        and row["result"].get("model_calls", 0) > 0 and row["record_id"] not in artifacts]
            infrastructure = [row for row in observed if row["result"].get("status") == "api_error"]
            parser_failures = [row for row in observed if row["record_id"] in artifacts]
            other = [row for row in observed if row not in complete and row not in infrastructure and row not in parser_failures]
            successes = [row for row in complete if row["result"].get("status") == "achieved"]
            summaries.append({
                "phase": phase, "family": family, "planned": len(group),
                "boards": len({case["layout_id"] for case in group}),
                "contexts": len({case["context_id"] for case in group}),
                "observed": len(observed), "completed": len(complete),
                "successes": len(successes), "success_fraction": len(successes) / len(complete) if complete else None,
                "infrastructure_failures": len(infrastructure), "other_unassessed": len(other),
                "parser_artifacts_unassessed": len(parser_failures),
                "not_started_or_unsaved": len(group) - len(observed),
                "invalid_responses": sum(row["result"].get("status") == "invalid_response" for row in complete),
                "goal_conjunction_intents": sum(case.get("effect_type") == "goal_conjunction" for case in group),
                "demonstration_offsets": json.dumps([case["k"] for case in group if "k" in case]),
                "successful_action_counts": json.dumps([len(row["result"]["actions"]) for row in successes]),
                "model_calls": sum(row["result"].get("model_calls", 0) for row in observed),
                "primitive_actions": sum(len(row["result"]["actions"]) for row in observed),
                "no_ops": sum(step.get("no_op") is True for row in observed
                              for step in row["result"].get("steps", [])),
                "attempt_seconds": sum(row.get("duration_seconds", 0) or 0 for row in observed),
            })
            cap_max = max((int(row.get("cap", 0)) for row in complete), default=0)
            for budget in range(1, cap_max + 1):
                eligible = [row for row in complete if int(row.get("cap", 0)) >= budget
                            or row["result"].get("status") in DECISIVE]
                items = [{"layout_id": row["layout_id"],
                          "success": row["result"].get("status") == "achieved"
                          and len(row["result"]["actions"]) <= budget} for row in eligible]
                lower, upper = board_interval(items)
                curve.append({"phase": phase, "family": family, "action_budget": budget,
                              "denominator": len(items), "successes": sum(row["success"] for row in items),
                              "fraction": sum(row["success"] for row in items) / len(items) if items else None,
                              "boards": len({row["layout_id"] for row in items}),
                              "bootstrap_low": lower, "bootstrap_high": upper})
    known_usage = [a["response"]["usage"] for a in audits
                   if isinstance(a.get("response"), dict) and isinstance(a["response"].get("usage"), dict)]
    usage = {"known_responses": len(known_usage), "without_usage": len(audits) - len(known_usage),
             "input_tokens": sum(row.get("input_tokens", 0) or 0 for row in known_usage),
             "output_tokens": sum(row.get("output_tokens", 0) or 0 for row in known_usage)}
    calibration = read_json(run / "results/calibration.json", {})
    try:
        ledger = read_ledger(audit_run)
    except (OSError, sqlite3.Error, ValueError) as exc:
        warnings.append(f"Cannot read shared budget ledger: {exc}")
        ledger = None
    return {"cases": cases, "results": results, "manifest": manifest, "plan": plan,
            "warnings": warnings, "audits": audits, "audit_run": audit_run, "audit_prefix": audit_prefix,
            "charge": settled_charge(audits), "campaign_charge": settled_charge(all_audits),
            "campaign_audit_count": len(all_audits),
            "campaign_audit_statuses": dict(Counter(audit.get("status") for audit in all_audits)), "token_usage": usage,
            "ledger": ledger,
            "parser_artifacts": artifacts,
            "summaries": summaries, "curve": curve,
            "calibration": calibration,
            "calibration_audit": audited_calibration(cases, results, artifacts, calibration)}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figures(run: Path, report: dict[str, Any]) -> list[Path]:
    """Save figures and their exact input tables without synthesising outcomes."""
    destination = run / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    curve = report["curve"]
    fields = ["phase", "family", "action_budget", "denominator", "successes", "fraction", "boards",
              "bootstrap_low", "bootstrap_high"]
    write_csv(destination / "success_by_action_budget.csv", curve, fields)
    summary_fields = ["phase", "family", "planned", "boards", "contexts", "observed", "completed", "successes",
                      "success_fraction", "infrastructure_failures", "other_unassessed", "not_started_or_unsaved",
                      "parser_artifacts_unassessed", "invalid_responses", "model_calls", "primitive_actions",
                      "no_ops", "attempt_seconds", "goal_conjunction_intents", "demonstration_offsets",
                      "successful_action_counts"]
    write_csv(destination / "summary.csv", report["summaries"], summary_fields)
    write_csv(destination / "calibration_audit.csv", report["calibration_audit"],
              ["context_id", "record_id", "status", "parser_artifact", "context_complete_after_audit", "successes",
               "success_fraction", "mean_success_actions", "u"])
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    colors = {FAMILIES[0]: "#2266a3", FAMILIES[1]: "#b34a36"}
    labels = {FAMILIES[0]: "12×12×4, dane realne", FAMILIES[1]: "12×12×10, dane syntetyczne"}
    for ax, phase in zip(axes, PHASES, strict=True):
        found = False
        for family in FAMILIES:
            rows = [row for row in curve if row["phase"] == phase and row["family"] == family
                    and row["denominator"]]
            if not rows:
                continue
            found = True
            x = [row["action_budget"] for row in rows]
            y = [row["fraction"] for row in rows]
            last = rows[-1]
            ax.step(x, y, where="post", marker=".", linewidth=1.4, color=colors[family],
                    label=f"{labels[family]} (n={last['denominator']}; plansze={last['boards']})")
            intervals = [row for row in rows if row["bootstrap_low"] is not None]
            if intervals:
                ax.vlines([r["action_budget"] for r in intervals], [r["bootstrap_low"] for r in intervals],
                          [r["bootstrap_high"] for r in intervals], color=colors[family], alpha=0.35)
        if found:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(0.5, 0.5, "Brak ukończonych prób", ha="center", va="center", transform=ax.transAxes)
        ax.set(title="Kalibracja" if phase == "calibration" else "Test", xlabel="Budżet ruchów gracza", ylim=(-0.03, 1.03))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Odsetek spełnionych intencji")
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = destination / f"success_by_action_budget.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)

    durations = [{"record_id": row["record_id"], "phase": row["phase"], "family": row["family"],
                  "layout_id": row["layout_id"], "successful_actions": len(row["result"]["actions"]),
                  "duration_seconds": row.get("duration_seconds")}
                 for row in report["results"].values()
                 if row["result"].get("status") == "achieved" and row["result"].get("model_calls", 0) > 0]
    write_csv(destination / "successful_durations.csv", durations,
              ["record_id", "phase", "family", "layout_id", "successful_actions", "duration_seconds"])
    if durations:
        fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
        ticks, tick_labels = [], []
        for phase in PHASES:
            for family in FAMILIES:
                group = [r for r in durations if r["phase"] == phase and r["family"] == family]
                if group:
                    position = len(ticks) + 1
                    ax.scatter([position] * len(group), [r["successful_actions"] for r in group],
                               color=colors[family], alpha=0.7)
                    ticks.append(position)
                    tick_labels.append(f"{phase}\n{labels[family]}\nn={len(group)}")
        ax.set_xticks(ticks, tick_labels, fontsize=8)
        ax.set_ylabel("Ruchy do pierwszego sukcesu")
        ax.grid(axis="y", alpha=0.2)
        for extension in ("png", "pdf", "svg"):
            path = destination / f"successful_durations.{extension}"
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    else:
        for extension in ("png", "pdf", "svg"):
            (destination / f"successful_durations.{extension}").unlink(missing_ok=True)

    variants = (("Pierwotny (parser)", report["audit_run"]), ("Ścisły JSON (parser)", report["audit_run"] / "structured"),
                ("Odpowiedź końcowa", report["audit_run"] / "final_response"),
                ("Powtórna ocena, 256", report["audit_run"] / "expanded"))
    overview = []
    for variant, variant_path in variants:
        if not (variant_path / "experiment_plan.json").exists():
            continue
        data = report if variant_path.resolve() == run.resolve() else load_report(variant_path)
        for family in FAMILIES:
            rows = [row for row in data["curve"] if row["phase"] == "test" and row["family"] == family]
            if rows:
                overview.append({"variant": variant, **max(rows, key=lambda row: row["action_budget"])})
    overview_fields = ["variant", *fields]
    write_csv(destination / "variant_overview.csv", overview, overview_fields)
    if overview:
        fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        labels_on_plot = [label for label, _ in variants if any(row["variant"] == label for row in overview)]
        for position, variant in enumerate(labels_on_plot):
            for offset, family in ((-0.16, FAMILIES[0]), (0.16, FAMILIES[1])):
                row = next((item for item in overview if item["variant"] == variant and item["family"] == family), None)
                if row is None or not row["denominator"]:
                    continue
                y = position + offset
                ax.scatter([row["fraction"]], [y], color=colors[family], s=35)
                if row["bootstrap_low"] is not None:
                    ax.hlines(y, row["bootstrap_low"], row["bootstrap_high"], color=colors[family], alpha=0.5)
                ax.annotate(f"{row['successes']}/{row['denominator']}", (row["fraction"], y),
                            xytext=(5, 2), textcoords="offset points", fontsize=8, color=colors[family])
        for family in FAMILIES:
            ax.scatter([], [], color=colors[family], label=labels[family])
        ax.set_yticks(range(len(labels_on_plot)), labels_on_plot)
        ax.set(xlim=(-0.03, 1.12), xlabel="Odsetek spełnionych intencji w dostępnym budżecie")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        for extension in ("png", "pdf", "svg"):
            path = destination / f"variant_overview.{extension}"
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    else:
        for extension in ("png", "pdf", "svg"):
            (destination / f"variant_overview.{extension}").unlink(missing_ok=True)
    return paths


def markdown(source: str) -> Any:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> Any:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def notebook(run_relative: str) -> Any:
    cells = [
        markdown("""
        # Luna w Sokobanie: kalibracja i wykonanie intencji

        Badamy, czy Luna potrafi wykonać zadaną intencję na planszy 12×12.
        Kalibracja oznacza tu **pomiar czasu działania tej samej zamrożonej polityki**
        na kilku intencjach w danym kontekście. Nie oznacza treningu ani strojenia wag Luny.
        Test przedstawiamy oddzielnie dla plansz z czterema i dziesięcioma skrzyniami.

        Notebook korzysta wyłącznie z zapisanych lokalnie danych i wyników; nie
        wysyła żądań API. Może przedstawiać wyniki częściowe — brak prób oraz
        awarie infrastruktury są podane oddzielnie i nie stają się porażkami modelu.
        Osobny raport Qwena, LoRA i ICL znajduje się w checkoutcie pod
        `analysis/intent_generator_20261114T032232Z/intent_generator_report.ipynb`.
        """),
        code(f'''
        from pathlib import Path
        import json
        import sys
        import pandas as pd
        from IPython.display import Code, Image, Markdown, display

        here = Path.cwd().resolve()
        ROOT = next((p for p in (here, *here.parents)
                     if (p / "pyproject.toml").exists() and (p / "carl").is_dir()), None)
        if ROOT is None:
            raise RuntimeError("Uruchom notebook w checkoutcie CaRL")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        RUN = ROOT / {run_relative!r}
        from scripts.build_luna_intent_notebook import figures, load_report, read_jsonl
        report = load_report(RUN)
        image_paths = figures(RUN, report)
        print("Checkout:", ROOT, "\\nWyniki:", RUN)
        print("Uwagi o kompletności:", report["warnings"] or "brak błędów odczytu")
        '''),
        markdown("""
        ## Dane i brak wycieku

        Plik `data/records.jsonl` zawiera wcześniej zamrożone przypadki. Każdy
        jest sprawdzoną lokalnie adnotacją na podstawie dwóch obserwacji rzeczywistej
        trajektorii. Późniejsza plansza i zapisane ruchy służą do audytu adnotacji,
        ale **nie są wejściem Luny**. Intencję opisują tablice `effects` i `preserve`:
        wszystkie warunki pierwszej mają być spełnione równocześnie, a warunki
        drugiej mają pozostawać prawdziwe przez cały przebieg. `instruction` jest
        pomijane przez kontroler. Każdy warunek z `effects` był fałszywy na początku
        fragmentu i prawdziwy w jego późniejszej obserwacji. Współrzędne
        `(wiersz, kolumna)` są zerowane.
        Skrzynie są nierozróżnialne. `k` to odstęp w demonstracji, nigdy czas Luny.

        Plansze z czterema skrzyniami pochodzą z realnej kohorty, a plansze z
        dziesięcioma z wcześniejszej syntetycznej kohorty proposal. To porównanie
        miesza liczbę skrzyń i źródło danych. Zachowany zapis wcześniejszej
        ekspozycji jest niepełny: pilotowe plansze nie muszą być całkowicie nowe.
        Określenie `fresh boards` w planie `final_response` oznacza jedynie
        rozłączność z testami pierwszych dwóch wariantów tej kampanii. Nie
        potwierdza, że model wcześniej nie widział tych plansz.
        Zależne intencje na tej samej planszy nie są niezależnymi planszami.
        """),
        code('''
        cases = report["cases"]
        data_table = pd.DataFrame([{
            "faza": r["phase"], "rodzina": r["family"], "plansza": r["layout_id"],
            "kontekst": r["context_id"], "źródło": r["source_path"], "SHA-256": r["source_sha256"],
            "pochodzenie intencji": r["annotation_provenance"]["kind"]
        } for r in cases])
        if not data_table.empty:
            display(data_table.groupby(["faza", "rodzina"]).agg(
                przypadki=("plansza", "size"), plansze=("plansza", "nunique"),
                konteksty=("kontekst", "nunique")))
            display(data_table.drop_duplicates(["faza", "rodzina", "plansza"]))
        else:
            print("Brak danych.")
        print("Pełny manifest danych:")
        display(Code(json.dumps(report["manifest"], ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        Niezależny audyt odtwarza zapisane ruchy Luny w środowisku, sprawdza
        stany po ruchach i zapisane warunki intencji. Poniższe skróty pozwalają
        stwierdzić, czy audyt dotyczy aktualnie zapisanych wyników. Nie zmienia
        on wyników ani nie uzupełnia pominiętych prób.
        """),
        code('''
        import hashlib
        from scripts.build_luna_intent_notebook import read_json
        audit = read_json(report["audit_run"] / "independent_rollout_audit.json", {})
        variant_dirs = {"v1": report["audit_run"], "structured": report["audit_run"] / "structured",
                        "final_response": report["audit_run"] / "final_response",
                        "expanded": report["audit_run"] / "expanded"}
        audit_checks = []
        for name, section in audit.get("variants", {}).items():
            directory = variant_dirs.get(name)
            if directory is None:
                audit_checks.append({"wariant": name, "plik": "nieznany wariant", "zgodny_sha256": False})
                continue
            for key, relative in (("records_jsonl", "data/records.jsonl"),
                                  ("rollouts_jsonl", "results/rollouts.jsonl"),
                                  ("experiment_plan_json", "experiment_plan.json")):
                path = directory / relative
                expected = section.get("input_sha256", {}).get(key)
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                audit_checks.append({"wariant": name, "plik": relative, "zgodny_sha256": actual == expected
                                     and actual is not None})
        if audit:
            print("Pełny zapis niezależnego audytu:")
            display(Code(json.dumps(audit, ensure_ascii=False, indent=2), language="json"))
            display(pd.DataFrame(audit_checks))
        else:
            print("Brak osobnego zapisanego audytu.")
        '''),
        markdown("""
        ## Sprzęt i czas

        Użytkownik podał komputer Dell XPS 15 9520, Core i9-12900HK, 64 GB RAM
        i RTX 3050 Laptop 4 GB, działający pod WSL. Te informacje opisują
        stanowisko, nie zakładaną szybkość. Wywołania Luny odbywają się przez API;
        plansze, checker i raport są przetwarzane lokalnie. Poniżej jest zapis
        etapów supervisora oraz suma czasów zapisanych prób — suma nie jest
        całkowitym czasem zegarowym, bo nie obejmuje przygotowania ani raportu.
        """),
        code('''
        import platform
        import psutil
        from scripts.build_luna_intent_notebook import read_json
        print("System:", platform.platform(), "Python:", platform.python_version(),
              "pamięć dostępna systemowi [GiB]:", round(psutil.virtual_memory().total / 2**30, 1))
        supervisor = read_json(report["audit_run"] / "run_manifest.json", {})
        print("Początek i deadline wspólnego eksperymentu:", supervisor.get("clock", {}))
        stages = [{key: stage.get(key) for key in ("name", "started_utc", "finished_utc", "status", "exit_code")}
                  for stage in supervisor.get("stages", [])]
        display(pd.DataFrame(stages) if stages else "Brak wpisów etapów supervisora")
        seconds = sum(row.get("duration_seconds", 0) or 0 for row in report["results"].values())
        print("Suma czasów zapisanych prób [s]:", round(seconds, 2))
        '''),
        markdown("""
        ## Procedura

        Model otrzymuje aktualną planszę, jej bieżące współrzędne, cel zadania,
        strukturę intencji oraz historię własnych ruchów i no-opów z tej próby.
        Zwraca jeden ruch `UP`, `DOWN`, `LEFT` albo `RIGHT`.
        Środowisko CaRL wykonuje ruch; po każdym kroku checker dokładnie sprawdza
        warunki intencji i zachowania. Liczymy także ruchy zablokowane, które nie
        zmieniają planszy. Próba zatrzymuje się po pierwszym spełnieniu intencji,
        naruszeniu `preserve`, rozwiązaniu planszy, błędzie lub wyczerpaniu budżetu.
        Dla tej samej polityki używamy tych samych reguł w kalibracji i teście.

        Jeśli istnieje co najmniej jeden sukces w danym kontekście, względną skalę
        `u` wyznaczamy z ważonego rozkładu średnich długości udanych prób;
        wagi są proporcjonalne do odsetków sukcesów, a remisy dostają środkowy
        percentyl. Zero sukcesów daje `u = null`. Zapisujemy `u` dopiero po
        ukończeniu pomiarów wszystkich kandydatów w danym kontekście. Jedna próba
        przy `temperature=0` oznacza mało dokładną skalę: odpowiedzi API nie muszą
        być ściśle deterministyczne. Nie tworzymy niezależnych powtórzeń przez
        ponowne odczytanie tej samej odpowiedzi. Przy jednym udanym kandydacie
        `u = 0,5` wynika z definicji i nie porównuje różnych długości.

        Pomiary kalibracyjne wykonano tylko na planszach 12×12×4. `u` ani
        przykłady z kalibracji nie są następnie przekazywane w testowym prompcie,
        a wagi Luny się nie zmieniają. Eksperyment nie sprawdza, czy użycie
        kalibracji poprawia politykę; nie przeprowadzono kalibracji na ×10.
        """),
        code('''
        print("Zamrożony plan eksperymentu:")
        display(Code(json.dumps(report["plan"], ensure_ascii=False, indent=2), language="json"))
        print("Wyniki kalibracji po audycie (u maskowane dla niekompletnych kontekstów):")
        display(pd.DataFrame(report["calibration_audit"]))
        calibration_contexts = {}
        for row in report["calibration_audit"]:
            calibration_contexts.setdefault(row["context_id"], []).append(row)
        complete_contexts = [rows for rows in calibration_contexts.values()
                             if rows and rows[0]["context_complete_after_audit"]]
        counts = [sum((row["successes"] or 0) > 0 for row in rows) for rows in complete_contexts]
        print("Kompletne konteksty:", len(complete_contexts), "| co najmniej dwa sukcesy:",
              sum(count >= 2 for count in counts), "| jeden sukces (u=0,5):", sum(count == 1 for count in counts),
              "| bez sukcesów:", sum(count == 0 for count in counts))
        print("Surowy zapis kalibracji dla przejrzystości; u z niekompletnych kontekstów nie jest uznawane:")
        display(Code(json.dumps(report["calibration"], ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        ## Wyniki z jawnymi mianownikami

        W mianowniku są ukończone próby, również pozostałe błędy formatu.
        W dwóch wcześniejszych wariantach wykryto błąd integracji: pole `output_text`
        mogło sklejać komentarz i końcową odpowiedź modelu. Jeżeli zapis API
        potwierdza, że właśnie tak powstała niepoprawna odpowiedź, a osobna
        odpowiedź końcowa jest poprawnym JSON-em z akcją, wyłączamy próbę z
        mianownika i pokazujemy ją jako błąd parsera. Nie wykonujemy tej akcji
        wstecznie i nie przypisujemy jej sukcesu. Pozostałe błędy formatu są
        liczone jako porażki formatu przy użytym limicie odpowiedzi.

        Awarie API, niepoprawne intencje, próby bez wywołania modelu i
        nieuruchomione przypadki podajemy osobno jako nieocenione. Wykluczenia
        mogą obciążyć wynik: odsetek ukończonych prób nie musi być reprezentatywny.
        Pionowe przedziały, gdy dostępne, to 95% bootstrap
        według plansz (2000 losowań); dla mniej niż pięciu plansz pokazujemy tylko
        opis. Punkty to obserwowane wartości dla kolejnych budżetów, bez interpolacji.
        """),
        code('''
        table = pd.DataFrame(report["summaries"])
        display(table if not table.empty else "Brak zaplanowanych przypadków")
        print("Skład intencji i czasy sukcesu (k opisuje demonstrację, nie trudność ani czas Luny):")
        for row in report["summaries"]:
            if row["phase"] == "test":
                print(row["family"], "koniunkcja z celem:", f"{row['goal_conjunction_intents']}/{row['planned']}",
                      "offsety demonstracji:", row["demonstration_offsets"],
                      "ruchy do sukcesu:", row["successful_action_counts"])
        print("Potwierdzone artefakty sklejenia odpowiedzi (bez wykonania akcji wstecznie):")
        artifacts = pd.DataFrame(report["parser_artifacts"].values())
        display(artifacts if not artifacts.empty else "brak potwierdzonych przypadków")
        display(Image(filename=str(RUN / "figures/success_by_action_budget.png")))
        print("Dane punktów i mianowniki:")
        curve = pd.DataFrame(report["curve"])
        display(curve if not curve.empty else "Brak ukończonych prób")
        duration_plot = RUN / "figures/successful_durations.png"
        if duration_plot.exists():
            display(Image(filename=str(duration_plot)))
            print("Czasy dotyczą tylko sukcesów; odsetki sukcesu podano osobno powyżej.")
        else:
            print("Brak zarejestrowanych sukcesów — nie rysujemy rozkładu czasów.")
        '''),
        markdown("""
        ### Przerwania i odrzucone odpowiedzi

        Zachowujemy pełne odpowiedzi, które przerwały próbę. Błąd sklejania
        rozpoznajemy tylko na podstawie zgodnego zapisu API i poprawnej osobnej
        odpowiedzi końcowej. Inne błędy formatu obejmują również odpowiedzi
        nieukończone w limicie tokenów; nie wyciągamy z nich wniosku o przyczynie
        rozumowania modelu. Szczegóły odpowiedzi końcowej są w audycie żądań.
        """),
        code('''
        failed = []
        for record_id, row in report["results"].items():
            result = row["result"]
            if result["status"] not in ("invalid_response", "api_error", "invalid_intent"):
                continue
            raw = result["steps"][-1].get("raw_response") if result.get("steps") else None
            failed.append({"record_id": record_id, "phase": row["phase"], "family": row["family"],
                           "status": result["status"], "parser_artifact": record_id in report["parser_artifacts"],
                           "error": result.get("error"), "raw_response": raw})
        display(pd.DataFrame(failed) if failed else "Brak zapisanych przerwań tego rodzaju")
        response_errors, response_error_warnings = read_jsonl(RUN / "results/response_errors.jsonl")
        if response_errors or response_error_warnings:
            print("Błędy odczytu odpowiedzi końcowej:", response_error_warnings)
            display(Code(json.dumps(response_errors, ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        ## Porównanie wariantów odpowiedzi

        Każdy z istniejących wariantów pokazujemy opisowo. Pierwsze dwa używały
        parsera, który mógł skleić komentarz z odpowiedzią końcową; nie wolno
        przypisywać wszystkich ich przerwań zachowaniu Luny. Plansze testowe
        pierwszych trzech wariantów są różne, a kalibracja może używać tych samych
        kontekstów. Wariant `expanded` ponownie ocenia plansze z wcześniejszych
        wariantów przy innym parserze i wyższym limicie tokenów: nie jest to nowy
        holdout. Tabela nie jest sparowanym testem statystycznym. Różnice mogą
        wynikać ze sposobu odczytu lub generowania odpowiedzi oraz z wyboru plansz.
        """),
        code('''
        campaign = report["audit_run"]
        variants = []
        for label, path in (("pierwotny", campaign), ("ścisły JSON", campaign / "structured"),
                            ("odpowiedź końcowa + ścisły JSON", campaign / "final_response"),
                            ("ponowna ocena, 256 tokenów", campaign / "expanded")):
            if (path / "experiment_plan.json").exists():
                variants.append((label, load_report(path)))
        if len(variants) > 1:
            comparison = [{"wariant": label, **row} for label, data in variants
                          for row in data["summaries"] if row["phase"] == "test"]
            display(pd.DataFrame(comparison))
            for i, (label, data) in enumerate(variants):
                layouts = {row["layout_id"] for row in data["cases"] if row["phase"] == "test"}
                for other_label, other in variants[i + 1:]:
                    other_layouts = {row["layout_id"] for row in other["cases"] if row["phase"] == "test"}
                    print(f"Wspólne plansze testowe: {label} / {other_label}: {len(layouts & other_layouts)}")
            print("Porównanie opisowe; zbiory mogą się pokrywać, brak sparowanego testu statystycznego.")
            overview_plot = RUN / "figures/variant_overview.png"
            if overview_plot.exists():
                display(Image(filename=str(overview_plot)))
        else:
            print("Brak kolejnych zapisanych wariantów.")
        '''),
        markdown("""
        ## Fragment do rozważenia w proposal

        Poniższy akapit jest wyłącznie opisem pomiarów; nie edytuje proposal.
        Pokaże się tylko wtedy, gdy w obu rodzinach ukończono co najmniej pięć prób.
        """),
        code('''
        groups = {r["family"]: r for r in report["summaries"] if r["phase"] == "test"}
        real = groups.get("real_12x12x4")
        synthetic = groups.get("proposal_gamma2_12x12x10")
        complete = lambda group: (group and group["completed"] >= 5 and not group["parser_artifacts_unassessed"]
                                  and not group["infrastructure_failures"] and not group["other_unassessed"]
                                  and not group["not_started_or_unsaved"])
        if complete(real) and complete(synthetic):
            setup = ("a repeat evaluation on previously tested layouts" if report["audit_prefix"] == "expanded:"
                     else "a small exploratory evaluation")
            print(f"In {setup}, a fixed Luna controller achieved "
                  f"{real['successes']}/{real['completed']} demonstration-derived intents on real "
                  f"12×12 boards with four boxes and {synthetic['successes']}/{synthetic['completed']} "
                  f"on synthetic boards with ten boxes. Conditions were checked on the actual "
                  f"environment states after each primitive action. The two cohorts differ in "
                  f"both source and box count, and prior exposure cannot be ruled out; these "
                  f"observations do not establish transfer or full-task performance.")
        else:
            print("Za mało ukończonych prób w obu rodzinach na sugerowany akapit.")
        '''),
        markdown("""
        ## Rzeczywiste przykłady

        Dla pierwszej dostępnej próby każdej fazy i rodziny pokazujemy początek,
        intencję, każdy ruch Luny oraz faktyczne plansze po ruchach. Plansza
        zarejestrowana później w demonstracji jest pokazana osobno jako źródło
        etykiety, **nie** jako wynik Luny. Przykłady nie są wybierane według sukcesu.

        Legenda: `#` ściana, `-` podłoga, `.` cel, `$` skrzynia, `*` skrzynia
        na celu, `@` gracz, `+` gracz na celu. Ruchy 0/1/2/3 to góra/dół/lewo/prawo.
        """),
        code('''
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        symbols = "#-.$*@+"
        palette = ListedColormap(["#29394b", "#f6f2e9", "#f2d77d", "#b87944", "#d1933e", "#6596c2", "#85b2c9"])
        def draw_board(ax, rows, title):
            values = np.array([[symbols.index(ch) for ch in row] for row in rows])
            ax.imshow(values, cmap=palette, vmin=-0.5, vmax=len(symbols)-0.5)
            for r, row in enumerate(rows):
                for c, char in enumerate(row):
                    if char in "$*@+":
                        ax.text(c, r, "P" if char in "@+" else "B", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(rows[0])))
            ax.set_yticks(range(len(rows)))
            ax.tick_params(axis="both", length=0, labelsize=5, pad=1)
            ax.xaxis.tick_top()
            ax.set_title(title, fontsize=9)

        chosen = {}
        for case in report["cases"]:
            key = (case["phase"], case["family"])
            if key not in chosen and case["record_id"] in report["results"]:
                chosen[key] = case
        if not chosen:
            print("Nie zapisano jeszcze ukończonego ani częściowego przebiegu.")
        for key, case in chosen.items():
            row = report["results"][case["record_id"]]
            result = row["result"]
            display(Markdown(f"### {key[0]} — {key[1]} — {case['record_id']}"))
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
            draw_board(axes[0], case["start_observation"], "Początek")
            draw_board(axes[1], result["endpoint"], "Faktyczny stan po próbie Luny")
            draw_board(axes[2], case["later_observation"], "Późniejsza demonstracja")
            example = RUN / "figures" / f"example_{case['record_id']}.png"
            fig.savefig(example, dpi=160)
            plt.close(fig)
            display(Image(filename=str(example)))
            print("Plansza początkowa:\\n" + "\\n".join(case["start_observation"]))
            print("Intencja:", json.dumps(case["intent"], ensure_ascii=False))
            for step in result["steps"]:
                print("\\nRuch:", step["action"], "no-op:", step["no_op"],
                      "intencja:", step["intent_holds"], "preserve:", step["preservation_holds"])
                print("Odpowiedź:", step["raw_response"])
                if step["after"] is not None:
                    print("\\n".join(step["after"]))
            print("Status:", result["status"], "liczba ruchów:", len(result["actions"]))
            if case["record_id"] in report["parser_artifacts"]:
                print("Uwaga: potwierdzony błąd sklejenia; finalna akcja nie została wykonana.")
            print("Późniejsza plansza z demonstracji — nie wynik Luny:\\n" + "\\n".join(case["later_observation"]))
        '''),
        markdown("""
        ## Pełne prompty i koszty

        Poniżej znajdują się kompletne obiekty każdego zapisanego żądania API,
        w tym prompt, wersja modelu, limit odpowiedzi i ustawienia. Odpowiedzi,
        identyfikatory i metryki zużycia pozostają we wspólnym `luna_requests/*.json`.
        Próbne wywołanie, jeżeli wystąpiło, jest zapisane oddzielnie i nie jest
        przypadkiem testowym. Limit kosztów obejmuje wszystkie wywołania w tym
        wspólnym rejestrze; poniżej pokazujemy zachowawczą kwotę rozliczonych
        żądań, która nie jest fakturą. Księga SQLite zachowuje dodatkową rezerwę
        dla prób o nieznanym lub nierozliczonym statusie. Rezerwa jest ochronnym
        limitem, **nie stwierdzonym wydatkiem**; nie zakładamy, że próba była
        bezpłatna. Całkowite obciążenie limitu obejmuje kwotę rozliczoną i rezerwę.
        """),
        code('''
        from collections import Counter
        print("Wspólny katalog audytu:", report["audit_run"])
        print("Prefiks tagów wariantu:", report["audit_prefix"] or "brak (bez tagów structured:, final: i expanded:)")
        print("Żądania wariantu według stanu:", dict(Counter(a.get("status") for a in report["audits"])))
        print("Zachowawcze rozliczenie wariantu [USD]:", report["charge"])
        print("Zachowawcze rozliczenie całej kampanii [USD]:", report["campaign_charge"])
        print("Liczba zapisanych żądań całej kampanii:", report["campaign_audit_count"])
        print("Stany żądań całej kampanii:", report["campaign_audit_statuses"])
        print("Znane zużycie tokenów wariantu:", report["token_usage"])
        print("Księga wspólnej kampanii, odczyt tylko do odczytu:")
        display(Code(json.dumps(report["ledger"], ensure_ascii=False, indent=2), language="json"))
        uncertain = [{key: audit.get(key) for key in ("reservation_id", "tag", "status", "error_type", "error_diagnostics")}
                     for audit in report["audits"] if audit.get("status") != "settled"]
        if uncertain:
            print("Nierozliczone żądania wariantu:")
            display(Code(json.dumps(uncertain, ensure_ascii=False, indent=2), language="json"))
        if not report["audits"]:
            print("Brak zapisanych wywołań API.")
        for audit in report["audits"]:
            print("\\nŻądanie:", audit.get("reservation_id"), "tag:", audit.get("tag"))
            display(Code(json.dumps(audit.get("request"), ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        ## Powtarzalność i ograniczenia

        Źródłowe wyniki to `results/rollouts.jsonl`, zdarzenia `results/events.jsonl`,
        pomiary kalibracji `results/calibration.json` i audyt `luna_requests/`.
        CSV oraz wykresy PNG, PDF i SVG są w `figures/`. Plan oraz manifest
        danych zawierają wybór przypadków, ścieżki i skróty. Do odtworzenia
        **samego raportu bez kosztów i bez zapytań API** z katalogu CaRL:

        Komendę dla tego wariantu drukuje ostatnia komórka.

        To małe, eksploracyjne badanie pojedynczej polityki, a nie dowód
        transferu, zdolności rozwiązywania pełnych plansz ani poprawy generatora
        LoRA. Intencje pochodzą z demonstracji i nie są predykcjami Qwena. Skala
        `u` jest relatywna do aktualnego kontrolera, zestawu kandydatów i budżetu;
        nie jest liczbą ruchów z demonstracji. Wyniki mogą być niekompletne z
        powodu deadline'u, błędów dostępu lub limitu kosztów — tabela pokazuje to
        bez uzupełniania braków.
        """),
        code('''
        import shlex
        print(".venv/bin/python scripts/build_luna_intent_notebook.py --run", shlex.quote(str(RUN)))
        '''),
    ]
    result = nbformat.v4.new_notebook(cells=cells)
    result.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    result.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    run_relative = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
    document = notebook(run_relative)
    # In-process execution also works in sandboxes that prohibit Jupyter's local sockets.
    shell = InteractiveShell.instance()
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        execution_count = 0
        for cell in document.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source)
            if result.error_before_exec or result.error_in_exec:
                raise result.error_before_exec or result.error_in_exec
            cell.execution_count = execution_count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for rich_output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=rich_output.data,
                                                           metadata=rich_output.metadata))
    finally:
        os.chdir(previous_directory)
    # Replace atomically, so an interruption does not erase an existing report.
    run.mkdir(parents=True, exist_ok=True)
    target = run / "luna_intent_report.ipynb"
    fd, temporary_name = tempfile.mkstemp(prefix=".notebook-", suffix=".ipynb", dir=run)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            nbformat.write(document, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(target)


if __name__ == "__main__":
    main()
