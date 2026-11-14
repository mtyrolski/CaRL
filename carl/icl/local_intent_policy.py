"""Run a frozen local language model as an intent-conditioned Sokoban policy.

Input JSONL records contain record_id, start_observation, task_goal and intent.
For generator comparisons, ``intent`` must be that generator's recorded output,
not the trajectory annotation. No demonstration endpoint or action is passed
to the policy. This module never trains or loads an adapter.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_rollout import rollout


@dataclass(frozen=True)
class PolicyConfig:
    """All decoding choices affecting a policy attempt."""

    model: str
    revision: str
    device: str = "cuda"
    max_new_tokens: int = 32
    max_context_tokens: int = 2048
    action_cap: int = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: PolicyConfig) -> str:
    payload = {**asdict(config), "prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
               "sampling": "greedy", "adapter": None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LocalIntentPolicy:
    """Generate one action from the fixed rollout prompt with frozen weights."""

    def __init__(self, tokenizer: Any, model: Any, torch: Any, config: PolicyConfig):
        if config.max_new_tokens < 1 or config.max_context_tokens < 1 or config.action_cap < 1:
            raise ValueError("Token limits and action cap must be positive")
        self.tokenizer, self.model, self.torch, self.config = tokenizer, model, torch, config
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, prompt: str) -> str:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.config.device)
        if ids.shape[1] + self.config.max_new_tokens > self.config.max_context_tokens:
            raise ValueError("Full policy prompt exceeds context limit; no truncation performed")
        pad_token = self.tokenizer.pad_token_id
        if pad_token is None:
            pad_token = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids),
                max_new_tokens=self.config.max_new_tokens, do_sample=False,
                pad_token_id=pad_token, use_cache=True,
            )
        return str(self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)).strip()


def run_record(record: dict[str, Any], decide: Any, action_cap: int) -> dict[str, Any]:
    """Evaluate an explicitly supplied intent without exposing other record fields."""
    state = from_rows(record["start_observation"])
    started = time.monotonic()
    result = rollout(environment(state), state, record["intent"], decide,
                     cap=action_cap, task_goal=record["task_goal"])
    return {"record_id": record["record_id"], "elapsed_seconds": time.monotonic() - started,
            "result": asdict(result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="Preselected JSONL with explicit intents")
    parser.add_argument("--record-id", action="append", help="Select fixed record ID; may be repeated")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--model", required=True, type=Path, help="Complete local model snapshot")
    parser.add_argument("--revision", required=True, help="Immutable source revision of local snapshot")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--action-cap", type=int, default=8)
    args = parser.parse_args()
    config = PolicyConfig(str(args.model.resolve()), args.revision, args.device, args.max_new_tokens,
                          args.max_context_tokens, args.action_cap)
    if not args.model.is_dir() or not (args.model / "model.safetensors").is_file():
        raise ValueError("A complete single-file local safetensors snapshot is required")
    if args.model.resolve().name != args.revision:
        raise ValueError("Model path must be the snapshot directory for the requested revision")
    if args.output.exists():
        raise FileExistsError("Use a new output directory for a frozen run")
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({row["record_id"] for row in records}) != len(records):
        raise ValueError("Records must be nonempty with unique IDs")
    if args.record_id:
        by_id = {row["record_id"]: row for row in records}
        if len(set(args.record_id)) != len(args.record_id) or any(key not in by_id for key in args.record_id):
            raise ValueError("Requested record IDs must be distinct and present")
        records = [by_id[key] for key in args.record_id]
    args.output.mkdir(parents=True)
    manifest = {"schema_version": 1, "config": asdict(config), "config_sha256": config_hash(config),
                "policy_prompt": POLICY_PROMPT, "records_sha256": sha256_file(args.records),
                "model_weights_sha256": sha256_file(args.model / "model.safetensors"),
                "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
                "command": sys.argv, "records": [row["record_id"] for row in records],
                "status": "prepared", "api_calls": 0,
                "error_interpretation": "api_error from rollout means local inference failure here"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype)
    model = model.to(args.device)
    decide = LocalIntentPolicy(tokenizer, model, torch, config)
    with (args.output / "results.jsonl").open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(run_record(record, decide, config.action_cap)) + "\n")
            output.flush()
            os.fsync(output.fileno())
    manifest["status"] = "complete"
    temporary = args.output / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "manifest.json")


if __name__ == "__main__":
    main()
