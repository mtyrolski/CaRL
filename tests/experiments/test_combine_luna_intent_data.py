"""Checks for complete, deterministic merging of frozen intent cases."""

import copy

import pytest

from scripts.combine_luna_intent_data import merge_records
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase


@pytest.fixture
def cohorts():
    calibration = [{"phase": Phase.CALIBRATION, "family": Family.REAL, "layout_id": f"cal-layout-{index // 3}",
                    "context_id": f"cal-context-{index // 3}", "record_id": f"cal-record-{index}"}
                   for index in range(12)]
    result = []
    for cohort, size in enumerate((8, 8, 4)):
        rows = copy.deepcopy(calibration)
        for family in Family:
            for index in range(size):
                identifier = f"{family}-{cohort}-{index}"
                rows.append({"phase": Phase.TEST, "family": family, "layout_id": f"layout-{identifier}",
                             "record_id": f"record-{identifier}"})
        result.append((f"source-{cohort}", rows))
    return result


def test_merge_includes_every_record_once_in_deterministic_order(cohorts):
    records, origins = merge_records(cohorts)
    assert len(records) == 52
    assert records[:12] == cohorts[0][1][:12]
    assert len(origins) == 40
    expected = {row["record_id"] for _, source in cohorts for row in source if row["phase"] == Phase.TEST}
    assert {row["record_id"] for row in records[12:]} == expected
    assert merge_records(list(reversed(cohorts)))[0] == records


def test_merge_rejects_a_test_layout_used_by_two_cohorts(cohorts):
    cohorts[1][1][12]["layout_id"] = cohorts[0][1][12]["layout_id"]
    with pytest.raises(ValueError, match="Repeated test layout"):
        merge_records(cohorts)


def test_merge_rejects_calibration_changes(cohorts):
    cohorts[1][1][0]["record_id"] = "altered"
    with pytest.raises(ValueError, match="Calibration records differ"):
        merge_records(cohorts)
