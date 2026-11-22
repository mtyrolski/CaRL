"""Exercise checked LoRA artifacts on a tiny, locally initialized CPU model."""

import json
import math
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from carl.icl.intent_annotations import prepare_annotations
from carl.icl.intents.generation.data import read_records
from carl.icl.policy_relative.measurements import MeasurementSettings, measure, trial_seed


REVISION = "c" * 40
RIGHT = '{"action":"RIGHT"}'
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' <eos> ' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
)


@dataclass(frozen=True)
class LocalFixture:
    """An independent base checkpoint and checked samples across three layouts."""

    model: Path
    demonstration: Path
    measured: Path
    queries: Path
    held_out: dict[str, Any]
    answer: str


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write(path: Path, content: Any) -> None:
    path.write_text(_compact(content) + "\n", encoding="utf-8")


def _tiny_qwen(path: Path, answer: str) -> None:
    import torch
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocabulary = {token: index for index, token in enumerate((
        "<unk>", "<eos>", "<pad>", "<assistant>", "<system>", "<user>", "<developer>", answer,
    ))}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    raw.add_tokens([AddedToken(answer, normalized=False)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
        additional_special_tokens=["<assistant>", "<system>", "<user>", "<developer>"],
        chat_template=CHAT_TEMPLATE,
    )
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
        bos_token_id=vocabulary["<system>"], eos_token_id=vocabulary["<eos>"],
        pad_token_id=vocabulary["<pad>"], tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        # A deterministic next token makes parser success testable. The attention
        # path stays active, so the training assertion can observe a LoRA update.
        embedding = model.model.embed_tokens.weight
        assistant_id, answer_id = vocabulary["<assistant>"], vocabulary[answer]
        direction = torch.zeros(config.hidden_size)
        direction[:2] = torch.tensor([0.6, -0.6])
        embedding[assistant_id].copy_(direction)
        embedding[answer_id].copy_(-direction)
        model.lm_head.weight.zero_()
        model.lm_head.weight[answer_id].copy_(0.4 * direction)
        model.lm_head.weight[vocabulary["<eos>"]].copy_(-0.4 * direction)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    _write(path / "experiment_revision.json", {"revision": REVISION})


def build_fixture(path: Path) -> LocalFixture:
    """Make independent checked D_A/D_C and a real Qwen3 checkpoint offline."""
    path.mkdir(parents=True, exist_ok=True)
    source = path / "demonstrations.json"
    trajectories = [
        {"source_id": split, "split": split, "task_group": f"group-{split}", "actions": actions,
         "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#", walls, "#-----#", "#######"]}
        for split, walls, actions in (
            ("train", "#-----#", [3, 3]),
            ("validation", "#--#--#", [3, 3]),
            ("test", "#-##--#", [3]),
        )
    ]
    _write(source, {"trajectories": trajectories})
    annotations = path / "annotations"
    result = prepare_annotations(source, annotations, max_offset=2, seed=11)
    assert result["accepted"] == 7
    demonstration = annotations / "D_A.jsonl"
    held_out, = read_records(demonstration, "test")
    output = path / "calibration"
    two_step = {row["record_id"]: 2 for row in read_records(demonstration)
                if row["split"] != "test" and row["demonstration_offset"] == 2}
    failed = {trial_seed(41, record_id, 1) for record_id in two_step}
    measurement = measure(
        demonstration, output, settings=MeasurementSettings(seed=41, action_cap=3, planned_attempts=two_step),
        controller={"id": "seeded-fixture-v1", "deterministic": False},
        policy_factory=lambda seed: lambda _prompt: '{"action":"LEFT"}' if seed in failed else RIGHT,
    )
    assert measurement["status"] == "complete"
    assert measurement["outcomes"] == {"action_limit": 2, "achieved": 6}
    queries = path / "queries.jsonl"
    _write(queries, {key: held_out[key] for key in (
        "record_id", "context_id", "layout_id", "split", "start_observation", "task_goal", "task_group",
    )})
    answer = _compact(held_out["intent"])
    model = path / "base"
    _tiny_qwen(model, answer)
    return LocalFixture(model, demonstration, output / "D_C.jsonl", queries, held_out, answer)


def _invoke(entry: Callable[[], None], *arguments: object) -> None:
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", [entry.__module__, *(str(argument) for argument in arguments)])
        try:
            entry()
        finally:
            for sig, previous in handlers.items():
                signal.signal(sig, previous)


def _runtime(fixture: LocalFixture) -> tuple[object, ...]:
    return ("--model", fixture.model, "--revision", REVISION, "--device", "cpu", "--dtype", "float32",
            "--threads", 1, "--max-length", 512, "--max-new-tokens", 2)


def _checkpoint(output: Path) -> tuple[Path, dict[str, Any]]:
    best = json.loads((output / "best.json").read_text(encoding="utf-8"))
    adapter = Path(best["adapter"])
    manifest = json.loads((adapter / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["step"] == best["step"] == 1
    assert math.isfinite(manifest["checkpoint"]["validation_loss"])
    assert len(manifest["chat_template_sha256"]) == 64
    assert manifest["role"] == "generator"
    assert manifest["base"]["revision"] == REVISION
    assert (adapter / "tokenizer/tokenizer.json").is_file()
    return adapter, manifest


@pytest.fixture(scope="module", name="tiny_data")
def fixture_tiny_data(tmp_path_factory: pytest.TempPathFactory) -> LocalFixture:
    pytest.importorskip("peft")
    return build_fixture(tmp_path_factory.mktemp("tiny-qwen"))


@pytest.fixture(scope="module", name="trained_adapter")
def fixture_trained_adapter(tiny_data: LocalFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from safetensors.torch import load_file

    from carl.icl.intent_generator_experiment import main

    folder = tmp_path_factory.mktemp("demo-lora")
    _invoke(main, "inspect", "--data", tiny_data.demonstration, "--output", folder / "inputs.json")
    output = folder / "run"
    _invoke(main, "train", "--data", tiny_data.demonstration, "--output", output, *_runtime(tiny_data),
            "--max-steps", 1, "--eval-every", 1, "--accumulation", 1, "--rank", 2,
            "--pad-to-length", 224, "--lora-target", "q_proj", "--lora-target", "v_proj")
    adapter, manifest = _checkpoint(output)
    assert manifest["prompt"]["scale_arm"] == "unscaled"
    weights = load_file(adapter / "adapter_model.safetensors", device="cpu")
    assert any(value.count_nonzero().item() for name, value in weights.items() if ".lora_B." in name)
    return adapter


def test_checked_demo_step_can_reload_as_local_cpu_role(
    tiny_data: LocalFixture, trained_adapter: Path, tmp_path: Path,
) -> None:
    from carl.icl.config import ExperimentConfig, LocalModelSettings
    from carl.icl.experiment import create_inference_api
    from carl.icl.intent_domain import SokobanObservation
    from carl.icl.intent_generator_experiment import main
    from carl.icl.llm_components.predicate import create_predicate_components
    from carl.icl.local_models import bind_local_role

    output = tmp_path / "evaluated.jsonl"
    _invoke(main, "evaluate", "--data", tiny_data.demonstration, "--output", output, *_runtime(tiny_data),
            "--adapter", trained_adapter)
    result, = read_records(output)
    assert result["split"] == "test" and result["json_valid"] is True
    assert result["parsed"] == tiny_data.held_out["intent"]

    kwargs = {"model": str(tiny_data.model), "revision": REVISION, "device": "cpu", "dtype": "float32"}
    local = LocalModelSettings.model_validate({**kwargs, "adapter": str(trained_adapter)})
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"), generator_tokens=128)
    observation = SokobanObservation(tuple(tiny_data.held_out["start_observation"]))
    results = []
    for _ in range(2):
        api = create_inference_api(config, "mock")
        components = create_predicate_components(
            config, api, context_limit=512, scale_arm="unscaled", local_roles={"generator": bind_local_role(local, "generator")},
        )
        results.append((components.propose(observation, tiny_data.held_out["task_goal"], None),
                        components.contract("generator")["fingerprint"]))
        assert components.metrics["local_generator_calls"] == 1
        assert components.metrics["physical_requests"] == 0
    assert results[0] == results[1]
    assert results[0][0] == tiny_data.held_out["intent"]

    api = create_inference_api(config, "mock")
    base_only = bind_local_role(LocalModelSettings.model_validate(kwargs), "generator")
    components = create_predicate_components(
        config, api, context_limit=512, scale_arm="unscaled", local_roles={"generator": base_only},
    )
    assert components.contract("generator")["fingerprint"] != results[0][1]


def test_actual_tokenizer_masks_prompt_and_padding(tiny_data: LocalFixture) -> None:
    import torch
    from transformers import AutoTokenizer

    from carl.icl.intents.generation.backend import tensor_example
    from carl.icl.intents.generation.data import encode_example

    tokenizer = AutoTokenizer.from_pretrained(tiny_data.model, local_files_only=True)
    record = next(row for row in read_records(tiny_data.demonstration, "train") if row["intent"] == tiny_data.held_out["intent"])
    encoded = encode_example(tokenizer, record, 512)
    tensors = tensor_example(torch, encoded, "cpu", pad_to_length=224, pad_token_id=tokenizer.pad_token_id)
    labels, attention = tensors["labels"][0], tensors["attention_mask"][0]
    supervised = labels[labels != -100].tolist()
    assert supervised == tokenizer.encode(tiny_data.answer + tokenizer.eos_token, add_special_tokens=False)
    assert ((labels == -100) & (attention == 1)).sum().item() > 0
    assert (attention == 0).sum().item() > 0
    assert torch.all(labels[attention == 0] == -100)


def test_checkpoint_rejects_mismatched_revision_role_and_modified_weights(
    tiny_data: LocalFixture, trained_adapter: Path, tmp_path: Path,
) -> None:
    from shutil import copytree

    from carl.icl.intents.generation.checkpoint import verify_adapter_artifact, verify_validation_selection

    with pytest.raises(ValueError, match="revision"):
        verify_adapter_artifact(trained_adapter, base=tiny_data.model, revision="e" * 40, role="generator")
    with pytest.raises(ValueError, match="role"):
        verify_adapter_artifact(trained_adapter, base=tiny_data.model, revision=REVISION, role="policy")
    relocated = tmp_path / "relocated"
    copytree(trained_adapter.parent, relocated)
    selected = relocated / trained_adapter.name
    moved_manifest = verify_adapter_artifact(selected, base=tiny_data.model, revision=REVISION, role="generator")
    verify_validation_selection(selected, moved_manifest)
    unselected = trained_adapter.parent / "checkpoint-last-1"
    manifest = verify_adapter_artifact(unselected, base=tiny_data.model, revision=REVISION, role="generator")
    with pytest.raises(ValueError, match="validation-selected"):
        verify_validation_selection(unselected, manifest)
    damaged = tmp_path / "damaged"
    copytree(trained_adapter, damaged)
    with (damaged / "adapter_model.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="pinned checkpoint"):
        verify_adapter_artifact(damaged, base=tiny_data.model, revision=REVISION, role="generator")


def test_measured_training_uses_pinned_adapter_on_target_free_test(
    tiny_data: LocalFixture, tmp_path: Path,
) -> None:
    from carl.icl.policy_relative.data import load_fit_data
    from carl.icl.policy_relative.types import PredictionStatus
    from carl.icl.policy_relative_lora import main

    output = tmp_path / "measured-lora"
    _invoke(main, "train", "--data", tiny_data.measured, "--output", output, "--arm", "scaled", *_runtime(tiny_data),
            "--min-informative-train-contexts", 1, "--max-steps", 1, "--eval-every", 1, "--accumulation", 2, "--rank", 2)
    adapter, manifest = _checkpoint(output)
    assert manifest["prompt"]["scale_arm"] == "relative_u"
    contexts = load_fit_data(tiny_data.measured).train
    pool = {candidate.record["record_id"]: (candidate, len(context.candidates))
            for context in contexts for candidate in context.candidates}
    sample_log, = read_records(output / "training_curve.jsonl")
    assert len(sample_log["samples"]) == 2
    assert any(sample["loss_multiplier"] != 1 for sample in sample_log["samples"])
    for sample in sample_log["samples"]:
        candidate, size = pool[sample["record_id"]]
        assert sample["scale"] == pytest.approx(candidate.u)
        assert sample["measured_weight"] == pytest.approx(candidate.weight)
        assert sample["loss_multiplier"] == pytest.approx(candidate.weight * size)
    output_file = tmp_path / "test.jsonl"
    _invoke(main, "evaluate", "--data", tiny_data.queries, "--output", output_file, "--arm", "scaled", *_runtime(tiny_data),
            "--adapter", adapter, "--requested-scale", 0.25)
    result, = read_records(output_file)
    assert result["status"] == PredictionStatus.VALID
    assert result["parsed"] == tiny_data.held_out["intent"]

    invalid = tmp_path / "not-test.jsonl"
    row, = read_records(tiny_data.queries)
    _write(invalid, {**row, "split": "train"})
    with pytest.raises(ValueError, match="split=test"):
        _invoke(main, "evaluate", "--data", invalid, "--output", tmp_path / "rejected.jsonl", "--arm", "scaled",
                *_runtime(tiny_data), "--adapter", adapter, "--requested-scale", 0.25)


def test_measured_weight_scales_only_supervised_token_gradient() -> None:
    import torch
    from torch.nn.functional import cross_entropy

    from carl.icl.policy_relative.training import _weighted_loss

    targets = torch.tensor([-100, 1, 2, -100])
    logits = torch.zeros((4, 3), requires_grad=True)
    baseline, = torch.autograd.grad(cross_entropy(logits, targets), logits)
    scaled, = torch.autograd.grad(_weighted_loss(cross_entropy(logits, targets), 2 / 3, 2), logits)
    assert baseline[targets != -100].count_nonzero().item() > 0
    assert scaled[targets == -100].count_nonzero().item() == 0
    torch.testing.assert_close(scaled, baseline / 3)
