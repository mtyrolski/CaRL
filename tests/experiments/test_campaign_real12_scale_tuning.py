"""Protect the smaller-board scope and balanced matched tuning contract."""
import json
from pathlib import Path

import pytest

from scripts.archive.campaign_20261102.tune_real12_campaign_scales import DESIGN, FAMILY, manifest_for, read_design


def test_design_rejects_large_board_and_unbalanced_grid(tmp_path: Path) -> None:
    design = read_design(DESIGN)
    assert design['families'] == [FAMILY]
    path = tmp_path / 'design.json'
    path.write_text(json.dumps({**design, 'families': [FAMILY, 'generated_32x32x10']}))
    with pytest.raises(ValueError, match='Only full real 12x12x4'):
        read_design(path)
    broken = {**design, 'scale_pairs': {**design['scale_pairs'], 'fixed': [[8, 4]]}}
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='two scale pairs'):
        read_design(path)


def test_manifest_keeps_board_identity_and_config_matched_across_methods() -> None:
    design = read_design(DESIGN)
    manifest = manifest_for(DESIGN, design, provider='openai', workers=1)
    assert manifest['cases'] == 12
    assert set(manifest['selected_board_ids']) == {FAMILY}
    boards = manifest['selected_board_ids'][FAMILY]
    assert len(boards) == 2 and len({board['id'] for board in boards}) == 2
    conditions = manifest['resolved_conditions']
    assert len(conditions) == 6
    assert {condition['method'] for condition in conditions} == {'relative_u', 'absolute_d', 'fixed'}
    assert all(condition['family'] == FAMILY and condition['split'] == 'scale_validation'
               and condition['action_budget'] == 128 and condition['repeat_seeds'] == [11]
               for condition in conditions)
    for key in ('limits', 'num_subgoals', 'max_expansions', 'json_max_retries'):
        assert all(condition[key] == conditions[0][key] for condition in conditions)
