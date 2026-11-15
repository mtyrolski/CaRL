"""Model-free checks for paired stage-four evaluation and baseline reuse."""
import hashlib
import json
import sys

import pytest

from carl.icl.intent_generator_experiment import prompt_messages
from scripts import overnight_stage04_evaluate as runner
from scripts.overnight_stage03_generate import TRANSFORMS
from scripts.overnight_stage03_generate import transform_rows


def contexts():
    result = []
    for family, source, index in (("12x12x4", "x4", 0), ("12x12x4", "x4", 1),
                                  ("12x12x10", "gamma", 0), ("12x12x10", "gamma", 1),
                                  ("12x12x10", "reverse", 0), ("12x12x10", "reverse", 1)):
        identity = f"{family}-{source}-{index}"
        board = ["############"] + ["#----------#"] * 10 + ["############"]
        result.append({"context_id": identity, "layout_id": f"layout-{identity}", "family": family,
                       "source_family": source, "split": "evaluation", "board_rank": index,
                       "original_rows": board, "source_hash": "sha", "adapter_training_exposure": False})
    return result


def reference_for(rows):
    cases, prompts = [], []
    for context in rows:
        for transform in TRANSFORMS:
            board = transform_rows(context["original_rows"], transform)
            messages = prompt_messages({"start_observation": board, "task_goal": runner.TASK_GOAL})
            rendered = json.dumps(messages, sort_keys=True)
            prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
            for method in ("base", "checkpoint220"):
                identifier = f"{context['context_id']}-{transform}-{method}"
                cases.append({**context, "case_id": identifier, "cohort_id": "old", "method": method,
                              "transform": str(transform), "source_rows": board,
                              "prompt_sha256": prompt_sha, "prompt_tokens": len(rendered.encode()[::4])})
                prompts.append({"case_id": identifier, "messages": messages,
                                "rendered_prompt": rendered, "prompt_sha256": prompt_sha})
    return {"cases": cases}, prompts


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        rendered = json.dumps(messages, sort_keys=True)
        return list(rendered.encode()[::4]) if tokenize else rendered

    def __call__(self, rendered, **_kwargs):
        return {"input_ids": list(rendered.encode()[::4])}


def test_selection_is_fixed_and_balanced_between_x10_sources():
    selected = runner.select_contexts(contexts(), 2)
    assert len(selected) == 4
    assert {row["source_family"] for row in selected if row["family"] == "12x12x10"} == {"gamma", "reverse"}
    with pytest.raises(ValueError, match="even"):
        runner.select_contexts(contexts(), 3)
    with pytest.raises(ValueError, match="repeated canonical"):
        runner.select_contexts(contexts() + [contexts()[0]], 2)


def test_split_separation_includes_validation_and_evaluation():
    selected = runner.select_contexts(contexts(), 2)
    layouts = [row["layout_id"] for row in selected]
    manifest = {"splits": {"train_layout_ids": ["train"], "validation_layout_ids": ["val"],
                           "evaluation_layout_ids": layouts}}
    runner.check_splits(selected, manifest)
    manifest["splits"]["train_layout_ids"].append(layouts[0])
    with pytest.raises(ValueError, match="not disjoint"):
        runner.check_splits(selected, manifest)


def test_freeze_enforces_exact_boards_and_declared_missing_arms():
    selected = runner.select_contexts(contexts(), 2)
    freeze = {"contexts": selected, "boards_per_family": 2, "canonical_original_boards": 4,
              "model_revision": runner.REVISION, "transforms": list(TRANSFORMS),
              "inference": {"device": "cpu", "dtype": "float32", "threads": 4, "batch_size": 3,
                            "decode": "greedy", "enable_thinking": False,
                            "max_length": 1024, "max_new_tokens": 128},
              "planned_methods": ["base", "historical_checkpoint220", "narrow", "wide", "wide_aug"]}
    assert runner.check_freeze(selected, freeze, 2) == freeze["planned_methods"]
    freeze["contexts"] = list(reversed(selected))
    with pytest.raises(ValueError, match="Selected boards"):
        runner.check_freeze(selected, freeze, 2)


def test_plan_uses_exact_same_permitted_prompt_for_every_arm():
    selected = runner.select_contexts(contexts(), 2)
    reference, original_prompts = reference_for(selected)
    plan, prompts = runner.build_plan(reference, original_prompts, selected,
                                      ["base", "historical_checkpoint220", "wide", "wide_aug"], FakeTokenizer())
    assert len(plan["cases"]) == 4 * 3 * 4
    assert len({case["case_id"] for case in plan["cases"]}) == len(plan["cases"])
    grouped = {}
    for case, prompt in zip(plan["cases"], prompts, strict=True):
        key = case["context_id"], case["transform"]
        grouped.setdefault(key, set()).add(prompt["rendered_prompt"])
        assert "future" not in json.dumps(prompt).lower()
        assert case["source_rows"] == transform_rows(case["original_rows"], case["transform"])
    assert all(len(group) == 1 for group in grouped.values())
    original_prompts[0]["messages"][-1]["content"] += "\nSECRET FUTURE"
    with pytest.raises(ValueError, match="current-board/goal"):
        runner.build_plan(reference, original_prompts, selected, ["base", "wide"], FakeTokenizer())


def test_only_valid_named_arms_are_accepted():
    name, path = runner.parse_arm("wide=/tmp/checkpoint")
    assert name == "wide" and str(path) == "/tmp/checkpoint"
    with pytest.raises(ValueError, match="Use --adapter"):
        runner.parse_arm("test=/tmp/checkpoint")


def test_recovery_keeps_missing_slots_separate_from_completed_outputs():
    cases = [{"case_id": "a"}, {"case_id": "b"}]
    status = runner.reconciled_status(cases, {"a": {"baseline_reused": True}}, ["base", "wide"])
    assert status["state"] == "partial" and status["completed"] == 1 and status["missing"] == 1
    assert status["baseline_reused"] == 1 and status["stop_reason"] == "reconciled_after_interruption"


def test_successful_reconciliation_of_partial_run_exits_successfully(monkeypatch):
    arguments = ["runner", "--reference", "reference", "--data-manifest", "data", "--freeze", "freeze",
                 "--output", "out", "--per-family", "2", "--resume"]
    monkeypatch.setattr(runner, "execute", lambda _args: {"state": "partial", "missing": 1})
    monkeypatch.setattr(sys, "argv", [*arguments, "--reconcile-only"])
    runner.main()
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1


def test_baselines_are_copied_only_for_matching_board_and_prompt():
    selected = runner.select_contexts(contexts(), 2)
    reference, original_prompts = reference_for(selected)
    methods = ["base", "historical_checkpoint220", "wide", "wide_aug"]
    plan, _ = runner.build_plan(reference, original_prompts, selected, methods, FakeTokenizer())
    previous = [{**case, "prediction_status": "complete", "raw_output": "not JSON",
                 "generated_tokens": 3, "generation_seconds": 2.5} for case in reference["cases"]]
    cached = runner.cached_baselines(previous, plan["cases"], "contract")
    assert len(cached) == 4 * 3 * 2
    assert all(row["raw_output"] == "not JSON" and row["baseline_reused"] for row in cached)
    assert {row["method"] for row in cached} == {"base", "historical_checkpoint220"}
    previous[0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="historical baseline|Historical baseline"):
        runner.cached_baselines(previous, plan["cases"], "contract")


def test_completed_matched_adapters_share_records_and_token_budget(tmp_path):
    data = tmp_path / "dataset.jsonl"
    rows = [{"record_id": "a", "layout_id": "train-a", "split": "train"},
            {"record_id": "b", "layout_id": "train-b", "split": "train"},
            {"record_id": "v", "layout_id": "val", "split": "validation"}]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"splits": {"train_layout_ids": ["train-a", "train-b"], "validation_layout_ids": ["val"]},
                "arms": {"wide": {"train_layout_ids": ["train-a", "train-b"]}},
                "output_sha256": {"dataset_wide.jsonl": runner.file_sha(data)}}
    arms = []
    for arm, augmentation in (("wide", "none"), ("wide_aug", "d4")):
        directory = tmp_path / arm
        checkpoint = directory / "checkpoint-2"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        adapter_config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4, "lora_alpha": 8,
                          "lora_dropout": 0.0, "bias": "none", "target_modules": ["q_proj", "v_proj"],
                          "base_model_name_or_path": str(runner.MODEL)}
        (checkpoint / "adapter_config.json").write_text(json.dumps(adapter_config))
        args = {key: None for key in runner.MATCHED_TRAINING_KEYS}
        args.update({"model": str(runner.MODEL), "revision": runner.REVISION, "max_steps": 2,
                     "data": str(data), "augmentation": augmentation})
        (directory / "configuration.json").write_text(json.dumps({"args": args, "data_sha256": runner.file_sha(data),
                                                                   "validation_record_ids": ["v"]}))
        (directory / "best.json").write_text(json.dumps({"step": 2, "validation_loss": .2, "adapter": str(checkpoint)}))
        (directory / "status.json").write_text(json.dumps({"status": "finished", "completed_steps": 2,
                                                            "best_adapter": str(checkpoint)}))
        curve = [{"step": step, "model_tokens": 20, "effective_tokens": 18, "supervised_tokens": 3,
                  "samples": [{"record_id": key, "layout_id": f"train-{key}", "symmetry": augmentation}]}
                 for step, key in ((1, "a"), (2, "b"))]
        curve[1]["validation_loss"] = .2
        (directory / "training_curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in curve))
        arms.append(f"{arm}={directory}")
    selected = runner.selected_adapters(arms, manifest)
    assert selected["wide"]["model_tokens"] == selected["wide_aug"]["model_tokens"] == 40
    adapter_path = tmp_path / "wide_aug/checkpoint-2/adapter_config.json"
    adapter_config = json.loads(adapter_path.read_text())
    adapter_config["r"] = 8
    adapter_path.write_text(json.dumps(adapter_config))
    with pytest.raises(ValueError, match="LoRA configuration"):
        runner.selected_adapters(arms, manifest)
    adapter_config["r"] = 4
    adapter_path.write_text(json.dumps(adapter_config))
    path = tmp_path / "wide_aug/training_curve.jsonl"
    curve = [json.loads(line) for line in path.read_text().splitlines()]
    curve[0]["samples"][0]["record_id"] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in curve))
    with pytest.raises(ValueError, match="different original records"):
        runner.selected_adapters(arms, manifest)
