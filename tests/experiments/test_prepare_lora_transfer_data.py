"""Transfer evaluation must keep layouts and future information separated."""

import copy
import random

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_generator_experiment import prompt_messages
from scripts.prepare_lora_transfer_data import alternating_ids
from scripts.prepare_lora_transfer_data import replay
from scripts.prepare_lora_transfer_data import select_contexts


def test_selection_is_balanced_and_independent_of_input_order():
    train = [{"record_id": f"tr{i}", "layout_id": f"train{i}", "split": "train"} for i in range(2)]
    test = [{"record_id": f"t{i:02d}-{j}", "layout_id": f"t{i:02d}", "split": "test"}
            for i in range(20) for j in range(2)]
    real = [{"record_id": f"r{i:02d}", "layout_id": f"r{i:02d}", "phase": "test",
             "split": "test", "family": "real_12x12x4"} for i in range(20)]
    selected = select_contexts(train + test, real)
    random.Random(10).shuffle(test)
    random.Random(11).shuffle(real)
    assert selected == select_contexts(train + test, real)
    assert len(selected) == len({row["layout_id"] for row in selected}) == 40
    assert all(row["split"] == "test" for row in selected)
    assert all(not row["record_id"].endswith("-1") for row in selected)
    identifiers = alternating_ids(selected)
    by_id = {row["record_id"]: row["evaluation_family"] for row in selected}
    for size in range(2, len(identifiers) + 1, 2):
        prefix = [by_id[identifier] for identifier in identifiers[:size]]
        assert prefix.count("real_12x12x4") == prefix.count("proposal_gamma2_12x12x10")


def test_selection_rejects_a_layout_exposed_in_training():
    test = [{"record_id": f"t{i}", "layout_id": f"layout{i}", "split": "test"} for i in range(20)]
    test.append({"record_id": "leak", "layout_id": "layout1", "split": "train"})
    real = [{"record_id": f"r{i}", "layout_id": f"real{i}", "phase": "test", "family": "real_12x12x4"}
            for i in range(20)]
    with pytest.raises(ValueError, match="disjoint"):
        select_contexts(test, real)


def test_replay_and_whitelisted_prompt_ignore_future_fields():
    start = ["############"] * 4 + ["#@$.-------#", "#$$$...----#"] + ["############"] * 6
    state = from_rows(start)
    later = to_rows(environment(state).next_state(state, 3))
    record = {"record_id": "secret-id", "layout_id": canonical_layout_id(state),
              "evaluation_family": "real_12x12x4", "start_observation": start, "later_observation": later,
              "fragment_actions": [3], "task_goal": "Put every box on a goal.",
              "intent": {"effects": [{"predicate": "box_at", "row": 4, "col": 3}], "preserve": []}}
    replay(record)
    changed = copy.deepcopy(record)
    changed.update(later_observation=["future-secret"], fragment_actions=[100], record_id="another-id")
    assert prompt_messages(record) == prompt_messages(changed)
    with pytest.raises(ValueError, match="next state"):
        replay({**record, "later_observation": start})
