"""Keep the proposed Figure 1 comparison paired and fully traceable."""

import csv
import json
from pathlib import Path

import pytest

from scripts import plot_figure1_cllp_matched as plotter


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def pilot(tmp_path):
    pairs, jobs, trials = [], [], []
    hits = {(arm, k): 0 for arm in ("luna", "terra_6_1") for k in plotter.OFFSETS}
    for board in range(plotter.BOARD_COUNT):
        for k in plotter.OFFSETS:
            pair = {"pair_id": f"pair-{board}-{k}", "layout_id": f"layout-{board}",
                    "context_id": f"context-{board}", "source_hash": f"source-{board}",
                    "target_hash": f"target-{board}-{k}", "k": k}
            pairs.append(pair)
            for arm, model in (("luna", "gpt-6-luna"), ("terra_6_1", "gpt-6.1-terra")):
                hit = board % 2 == 0
                hits[(arm, k)] += hit
                key = f"trial-{arm}-{board}-{k}"
                job = {"key": key, "arm": arm, "context": pair["context_id"], "layout": pair["layout_id"],
                       "pair": pair["pair_id"], "source_hash": pair["source_hash"], "model": model,
                       "target_hash": pair["target_hash"], "k": k, "repetition": 0, "group": "main"}
                jobs.append(job)
                trace_path = tmp_path / f"{key}.json"
                trace_path.write_text(json.dumps({"identity": job, "status": "complete", "exact_hit": hit,
                    "first_hit": 1 if hit else None, "attempted_actions": 1,
                    "endpoint_hash": pair["target_hash"] if hit else pair["source_hash"]}), encoding="utf-8")
                trials.append({"family": "12", "arm": arm, "layout": pair["layout_id"],
                               "context": pair["context_id"], "k": k, "trial_key": key, "status": "complete",
                               "hit": int(hit), "first_hit": 1 if hit else "", "attempted_actions": 1,
                               "trace_path": str(trace_path)})
    paths = {name: tmp_path / name for name in ("pairs.json", "trials.csv", "summary.csv", "manifest.json",
                                                "verification.json")}
    paths["pairs.json"].write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    write_csv(paths["trials.csv"], trials)
    summary = [{"family": "12", "arm": arm, "k": k, "status": "COMPLETE", "resolved": 20, "planned": 20,
                "hits": hits[(arm, k)]} for arm in ("luna", "terra_6_1") for k in plotter.OFFSETS]
    write_csv(paths["summary.csv"], summary)
    manifest = {"dataset_artifacts": {"core_pairs.json": plotter.sha256(paths["pairs.json"])}, "jobs": jobs,
                "study": {"core_offsets": list(plotter.OFFSETS), "core_repetitions": 1, "action_cap": 32,
                          "few_shot": 32, "policy_temperature": 1.0}}
    paths["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["verification.json"].write_text(json.dumps({"12": {"mode": "live", "counts": {"complete": 240},
                                                            "verified_files": 240}}), encoding="utf-8")
    return paths


def load(pilot):
    return plotter.load_baseline(pilot["pairs.json"], pilot["trials.csv"], pilot["summary.csv"],
                                 pilot["manifest.json"], pilot["verification.json"])


def cllp_rows(pairs, model):
    rows = []
    for pair in pairs.values():
        hit = int(pair.layout_id.split("-")[-1]) % 2 == 0
        actions = [0] if hit else [0] * 32
        hashes = [pair.source_hash, pair.target_hash] if hit else [pair.source_hash] * 33
        rows.append({"model": model, "pair_id": pair.pair_id, "layout_id": pair.layout_id,
                     "source_hash": pair.source_hash, "target_hash": pair.target_hash, "k": pair.k,
                     "action_cap": 32, "exact_hit": hit, "first_hit": 1 if hit else None,
                     "actions": actions, "state_hashes": hashes, "no_op_count": 0 if hit else 32,
                     "attempted_actions": len(actions), "stop_reason": "exact_target" if hit else "action_limit"})
    return rows


def test_pilot_joins_original_trace_and_manifest(pilot):
    pairs, results = load(pilot)
    assert len(pairs) == 120
    assert len(results[plotter.Model.LUNA]) == len(results[plotter.Model.TERRA]) == 120
    assert all(sum(results[plotter.Model.LUNA][key] for key, pair in pairs.items() if pair.k == k) == 10
               for k in plotter.OFFSETS)


def test_pilot_rejects_altered_trace_identity(pilot):
    row = plotter.read_csv(pilot["trials.csv"])[0]
    path = Path(row["trace_path"])
    trace = json.loads(path.read_text())
    trace["identity"]["target_hash"] = "changed"
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="trace identity"):
        load(pilot)


def test_pilot_rejects_missing_trial_and_bad_summary(pilot):
    rows = plotter.read_csv(pilot["trials.csv"])
    write_csv(pilot["trials.csv"], rows[:-1])
    with pytest.raises(ValueError, match="Missing paired"):
        load(pilot)


@pytest.mark.parametrize("change", ["wrong_hash", "wrong_model", "no_op_count", "action_cap", "missing", "duplicate"])
def test_cllp_requires_complete_unmodified_pairs(pilot, tmp_path, change):
    pairs, _ = load(pilot)
    rows = cllp_rows(pairs, plotter.Model.CLLP_8)
    if change == "wrong_hash":
        rows[0]["target_hash"] = "other"
    elif change == "wrong_model":
        rows[0]["model"] = plotter.Model.CLLP_32
    elif change == "no_op_count":
        rows[0]["no_op_count"] = 99
    elif change == "action_cap":
        rows[0]["action_cap"] = 16
    elif change == "missing":
        rows.pop()
    else:
        rows.append(rows[0])
    path = tmp_path / "cllp.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        plotter.load_cllp(path, plotter.Model.CLLP_8, pairs)


def test_matched_summary_and_paired_bootstrap(pilot, tmp_path):
    pairs, results = load(pilot)
    for model in (plotter.Model.CLLP_8, plotter.Model.CLLP_32):
        path = tmp_path / f"{model}.jsonl"
        write_jsonl(path, cllp_rows(pairs, model))
        results[model] = plotter.load_cllp(path, model, pairs)
    rows = plotter.summarise(pairs, results)
    assert len(rows) == 24 and all(row["hits"] == 10 and row["boards"] == 20 for row in rows)
    differences = plotter.paired_differences(pairs, results, samples=100)
    assert len(differences) == 24
    assert all(row["difference"] == row["bootstrap_low"] == row["bootstrap_high"] == 0 for row in differences)
    assert plotter.wilson(0, 20)[1] > 0
    assert plotter.wilson(20, 20)[0] < 1


def test_rejects_incomplete_checkpoint_run_and_changed_weights(pilot, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "pytorch_model.bin"):
        (checkpoint / name).write_text(name)
    out = tmp_path / "run"
    out.mkdir()
    trial_path = out / "trials.jsonl"
    trial_path.write_text("")
    hashes = {str(path.resolve()): plotter.sha256(path) for path in
              (pilot["pairs.json"], checkpoint / "config.json", checkpoint / "pytorch_model.bin")}
    (out / "provenance.json").write_text(json.dumps({"model": "CLLP-8", "checkpoint": str(checkpoint),
        "action_cap": 32, "token_length": 291, "policy": "greedy argmax", "sha256": hashes}))
    status = out / "status.json"
    status.write_text(json.dumps({"state": "running", "completed": 119, "requested": 120}))
    with pytest.raises(ValueError, match="Incomplete"):
        plotter.validate_cllp_artifacts(trial_path, plotter.Model.CLLP_8, pilot["pairs.json"])
    status.write_text(json.dumps({"state": "complete", "completed": 120, "requested": 120}))
    assert len(plotter.validate_cllp_artifacts(trial_path, plotter.Model.CLLP_8, pilot["pairs.json"])) == 2
    (checkpoint / "pytorch_model.bin").write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        plotter.validate_cllp_artifacts(trial_path, plotter.Model.CLLP_8, pilot["pairs.json"])
