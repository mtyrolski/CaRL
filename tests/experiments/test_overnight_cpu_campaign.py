"""Checks for the persistent local campaign clock and active pointer."""

import json

import pytest

from scripts.overnight_cpu_campaign import remaining_seconds
from scripts.overnight_cpu_campaign import status


def test_remaining_uses_earlier_monotonic_and_utc_deadlines():
    clock = {"deadline_epoch": 200.0, "deadline_monotonic": 80.0}
    assert remaining_seconds(clock, now_epoch=100.0, now_monotonic=50.0, same_boot=True) == 30.0
    assert remaining_seconds(clock, now_epoch=100.0, now_monotonic=50.0, same_boot=False) == 100.0
    assert remaining_seconds(clock, now_epoch=201.0, now_monotonic=50.0, same_boot=True) == 0.0


def test_status_preserves_manifest_and_observes_supervisor(tmp_path):
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_text(json.dumps({"status": "deadline_reached"}))
    manifest = tmp_path / "campaign.json"
    value = {"campaign_id": "run", "boot_id": "other-boot",
             "clock": {"deadline_epoch": 9999999999, "deadline_monotonic": 1, "deadline_utc": "deadline"},
             "stages": {"01": {"status": "running", "supervisor_manifest": str(supervisor)}}}
    original = json.dumps(value)
    manifest.write_text(original)
    pointer = tmp_path / "active.json"
    pointer.write_text(json.dumps({"campaign_id": "run", "manifest": str(manifest),
                                   "global_deadline_utc": "deadline"}))
    result = status(pointer)
    assert result["stages"]["01"]["status"] == "running"
    assert result["stages"]["01"]["supervisor_status"] == "deadline_reached"
    assert not result["monotonic_clock_valid"]
    assert manifest.read_text() == original


def test_status_rejects_pointer_to_different_campaign(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"campaign_id": "other"}))
    pointer = tmp_path / "active.json"
    pointer.write_text(json.dumps({"campaign_id": "run", "manifest": str(manifest)}))
    with pytest.raises(ValueError, match="does not match"):
        status(pointer)
