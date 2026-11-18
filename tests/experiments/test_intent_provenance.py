"""Source manifests must bind the implementation used by each experiment."""

import hashlib

import pytest

from carl.icl.intents.provenance import component_hashes


def test_component_hashes_track_each_file_without_mutation(tmp_path):
    (tmp_path / "nested").mkdir()
    source = tmp_path / "nested" / "model.py"
    source.write_bytes(b"original")

    recorded = component_hashes(("nested/model.py",), root=tmp_path)
    assert recorded == {"nested/model.py": hashlib.sha256(b"original").hexdigest()}
    assert source.read_bytes() == b"original"
    source.write_bytes(b"changed")
    assert component_hashes(("nested/model.py",), root=tmp_path) != recorded


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py"])
def test_component_hashes_reject_paths_outside_the_package(tmp_path, name):
    with pytest.raises(ValueError, match="relative source path"):
        component_hashes((name,), root=tmp_path)
