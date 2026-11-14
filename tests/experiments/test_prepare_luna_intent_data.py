"""Checks for shared calibration contexts and frozen test selection."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent
from scripts.audit_luna_intent_data import audit
from scripts.audit_luna_intent_data import validate_reference_cohorts
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase
from scripts.prepare_luna_intent_data import choose_context
from scripts.prepare_luna_intent_data import make_records
from scripts.prepare_luna_intent_data import read_reference_cohort
from scripts.prepare_luna_intent_data import write_frozen


@pytest.fixture
def trajectory():
    rows = ["#######", "#@-$-.#", "#--$-.#", "#-----#", "#-----#", "#-----#", "#######"]
    state = from_rows(rows)
    env = environment(state)
    actions = [3, 3, 3, 1]
    states = [rows]
    for action in actions:
        state = env.next_state(state, action)
        states.append(to_rows(state))
    return states, actions


def test_calibration_candidates_share_start_and_are_distinct(trajectory):
    states, actions = trajectory
    source = {"layout_id": "layout", "trajectory_id": "trajectory", "source_path": "source",
              "source_sha256": "hash"}
    records = make_records(source, Phase.CALIBRATION, Family.REAL, states, actions, 2, {})
    assert len(records) == 2
    assert len({record["context_id"] for record in records}) == 1
    assert all(record["start_observation"] == states[0] for record in records)
    assert len({json.dumps(record["intent"], sort_keys=True) for record in records}) == 2
    for record in records:
        assert not validate_intent(record["intent"], record["fragment_observations"])
        assert record["fragment_actions"] == actions[:record["k"]]
        assert record["policy_measurements"] is None


def test_test_offsets_pick_nearest_valid_and_break_ties_downward(trajectory):
    states, _ = trajectory
    assert choose_context(states, 1, target_offset=1)[1][0][0] == 2
    assert choose_context(states, 1, target_offset=3)[1][0][0] == 2
    assert choose_context(states, 1, target_offset=4)[1][0][0] == 4


def test_existing_frozen_cohort_cannot_be_replaced(tmp_path):
    records = [{"record_id": "one"}]
    write_frozen(tmp_path, records, [], {"version": 1})
    write_frozen(tmp_path, records, [], {"version": 1})
    with pytest.raises(ValueError, match="Refusing to replace different frozen data"):
        write_frozen(tmp_path, [{"record_id": "two"}], [], {"version": 1})
    assert json.loads((tmp_path / "records.jsonl").read_text()) == records[0]


def test_reference_cohort_reads_only_frozen_inputs_and_detects_inconsistency(tmp_path):
    records = [{"layout_id": "a", "phase": "calibration"}, {"layout_id": "b", "phase": "test"},
               {"layout_id": "c", "phase": "test"}]
    manifest = {"calibration_layouts": ["a"], "test_layouts_x4": ["b"], "test_layouts_x10": ["c"],
                "record_count": 3}
    write_frozen(tmp_path, records, [], manifest)
    layouts, calibration = read_reference_cohort(tmp_path / "manifest.json")
    assert layouts == {"a", "b", "c"}
    assert calibration == records[:1]
    manifest["test_layouts_x10"] = ["other"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest and frozen records disagree"):
        read_reference_cohort(tmp_path / "manifest.json")


def test_independent_audit_rejects_an_altered_action():
    rows = ["############", "#@-$-.-----#", "#--$-.-----#", "#--$-.-----#", "#--$-.-----#",
            "#----------#", "#----------#", "#----------#", "#----------#", "#----------#",
            "#----------#", "############"]
    initial = from_rows(rows)
    env = environment(initial)
    actions = [3, 3, 3, 1]
    states = [rows]
    state = initial
    for action in actions:
        state = env.next_state(state, action)
        states.append(to_rows(state))
    source = {"layout_id": canonical_layout_id(initial), "trajectory_id": "trajectory",
              "source_path": "source", "source_sha256": "hash"}
    record = make_records(source, Phase.CALIBRATION, Family.REAL, states, actions, 1, {})[0]
    record["fragment_actions"][0] = 2
    with pytest.raises(ValueError, match="Recorded action does not produce"):
        audit([record], [])


def test_every_reference_is_checked_for_test_layout_overlap():
    calibration = [{"phase": "calibration", "layout_id": "cal"}]
    first = calibration + [{"phase": "test", "layout_id": "old-one"}]
    second = calibration + [{"phase": "test", "layout_id": "old-two"}]
    validate_reference_cohorts(calibration, {"new"}, [first, second])
    with pytest.raises(ValueError, match="overlaps a reference cohort"):
        validate_reference_cohorts(calibration, {"old-two"}, [first, second])
