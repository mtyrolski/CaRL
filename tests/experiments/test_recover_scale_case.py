"""A retry preserves and checks observed history instead of hiding failed work."""
from pathlib import Path

import pytest

from scripts.archive.campaign_20261102.recover_scale_case import save_immutable
from scripts.archive.campaign_20261102.resume_calibration_v2 import invalid_policy_output


def test_rejects_completed_policy_response_with_unclosed_array() -> None:
    record = {'status': 'completed', 'request': {'payload': {'metadata': {'component': 'policy'}}},
              'output_text': '{"probabilities":[0.0,1.0,0.0,0.0}'}
    assert invalid_policy_output(record)
    record['output_text'] = '{"probabilities":[0.0,1.0,0.0,0.0]}'
    assert not invalid_policy_output(record)


def test_immutable_archive_refuses_overwriting_evidence(tmp_path: Path) -> None:
    target = tmp_path / 'evidence.json'
    save_immutable(target, b'original')
    save_immutable(target, b'original')
    with pytest.raises(ValueError, match='Archive differs'):
        save_immutable(target, b'replacement')
    assert target.read_bytes() == b'original'
