"""Offline token accounting and clearly bounded scenario assumptions."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from carl.icl.budget_sweep import load_sweep
from carl.icl.token_estimate import Assumptions
from carl.icl.token_estimate import estimate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_token_estimate_data(monkeypatch):
    """Use real board/prompt construction without an external trajectory dataset."""
    from carl.icl.boards import from_rows
    from carl.icl.boards import to_rows
    from carl.icl.datasets import environment

    interior = ['----------'] * 10
    interior[0] = '@---------'
    interior[4] = '-$-$-$-$--'
    interior[8] = '-.-.-.-.--'
    rows = ['############', *(f'#{row}#' for row in interior), '############']
    initial = from_rows(rows)
    env = environment(initial)
    states = [initial]
    for _ in range(8):
        states.append(env.next_state(states[-1], 3))
    candidates = [
        {'k': k, 'actions': [3] * k, 'subgoal': to_rows(states[k])}
        for k in (1, 2, 4, 8, 1, 2, 4, 8)
    ]
    training = {'contexts': [
        {'id': f'train-{i}', 'state': rows, 'remaining_actions': 8,
         'candidates': candidates, 'calibration_indices': list(range(8))}
        for i in range(8)
    ]}
    instances = [{'state': rows}, {'state': to_rows(states[1])}]
    monkeypatch.setattr('carl.icl.token_estimate.load_data', lambda _config: (training, instances))


@pytest.fixture
def cached_o200k_encoding():
    """Run the real tokenizer test only with a complete local encoding cache."""
    url = 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
    digest = '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'
    cache_dir = os.environ.get('TIKTOKEN_CACHE_DIR', os.environ.get(
        'DATA_GYM_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'data-gym-cache')
    ))
    if not cache_dir:
        pytest.skip('o200k_base encoding cache is disabled')
    cache_file = Path(cache_dir) / hashlib.sha1(url.encode()).hexdigest()
    if not cache_file.is_file() or hashlib.sha256(cache_file.read_bytes()).hexdigest() != digest:
        pytest.skip('o200k_base encoding is not available in the local cache')


def assert_quick_token_estimate(monkeypatch):
    from carl.icl import experiment
    from carl.icl.api import CachedOpenAI

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


@pytest.mark.usefixtures('synthetic_token_estimate_data', 'deterministic_token_estimates')
def test_quick_token_estimate_accounts_budget_without_api_or_dotenv(monkeypatch):
    assert_quick_token_estimate(monkeypatch)


@pytest.mark.usefixtures('synthetic_token_estimate_data', 'cached_o200k_encoding')
def test_quick_token_estimate_uses_real_tokenizer_without_api_or_dotenv(monkeypatch):
    assert_quick_token_estimate(monkeypatch)


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
