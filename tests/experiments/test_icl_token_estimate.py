"""Offline token accounting and clearly bounded scenario assumptions."""
import json
from pathlib import Path

import pytest

from carl.icl.budget_sweep import load_sweep
from carl.icl.token_estimate import Assumptions, estimate

ROOT = Path(__file__).resolve().parents[2]


def test_quick_token_estimate_uses_real_prompts_without_api_or_dotenv(monkeypatch):
    from carl.icl.api import CachedOpenAI
    from carl.icl import experiment

    def forbidden(*_args, **_kwargs):
        pytest.fail("Token estimation must not access the API or dotenv")

    monkeypatch.setattr(CachedOpenAI, "_client", forbidden)
    monkeypatch.setattr(experiment, "load_dotenv", forbidden)
    spec, configs = load_sweep("configs/icl/budget_sweeps/quick.json")
    assumptions = Assumptions.model_validate_json(
        (ROOT / "configs/icl/token_estimates/assumptions.json").read_text()
    )
    result = estimate(spec, configs, assumptions)
    assert result["plan"]["runs"] == 8
    assert result["logical_limits"]["search_input_proxy"] == 8_000_000
    assert result["logical_limits"]["search_output_tokens"] == 8 * 32768
    assert result["logical_limits"]["calibration_max_policy_calls"] == 4096
    assert all(sample["generator"]["mean"] > 0 for sample in result["samples"].values())
    assert (
        len({sample["calibration_signature"] for sample in result["samples"].values()})
        == 1
    )
    middle = next(row for row in result["scenarios"] if row["scenario"] == "middle")
    assert middle["calibration_calls"] == 2048
    assert middle["total_input"] == middle["calibration_input"] + middle["search_input"]
    assert (
        middle["total_output"] == middle["calibration_output"] + middle["search_output"]
    )
    assert middle["estimated_usd_uncached"] == pytest.approx(
        (middle["total_input"] * 0.1 + middle["total_output"] * 0.5) / 1_000_000,
        abs=0.0002,
    )


def test_price_model_and_invalid_assumptions_are_rejected():
    spec, configs = load_sweep("configs/icl/budget_sweeps/quick.json")
    raw = json.loads(
        (ROOT / "configs/icl/token_estimates/assumptions.json").read_text()
    )
    raw["pricing"]["model"] = "other-model"
    with pytest.raises(ValueError, match="Price model"):
        estimate(spec, configs, Assumptions.model_validate(raw))
    raw["scenarios"]["middle"]["action_fraction"] = 2
    with pytest.raises(ValueError):
        Assumptions.model_validate(raw)


def test_multiple_proposal_estimates_are_rejected():
    spec, configs = load_sweep("configs/icl/budget_sweeps/quick.json")
    configs[0].num_subgoals = 2
    raw = json.loads(
        (ROOT / "configs/icl/token_estimates/assumptions.json").read_text()
    )
    with pytest.raises(ValueError, match="num_subgoals=1"):
        estimate(spec, configs, Assumptions.model_validate(raw))
