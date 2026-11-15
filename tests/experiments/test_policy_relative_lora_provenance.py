"""Exercise the policy-relative entry point and source provenance."""

import hashlib
import json
import subprocess
import sys

from carl.icl.policy_relative import provenance


def test_provenance_covers_entry_point_package_and_shared_modules(
    tmp_path, monkeypatch
):
    root = tmp_path / "icl"
    package = root / "policy_relative"
    package.mkdir(parents=True)
    facade = root / "policy_relative_lora.py"
    facade.write_text("entry point", encoding="utf-8")
    for name in ("__init__.py", "provenance.py", "training.py"):
        (package / name).write_text(name, encoding="utf-8")
    for name in provenance.SHARED_CODE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    original, files = provenance.code_provenance()
    assert set(files) == set(provenance.SHARED_CODE_FILES) | {
        "policy_relative_lora.py",
        "policy_relative/__init__.py",
        "policy_relative/provenance.py",
        "policy_relative/training.py",
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    assert original == hashlib.sha256(payload).hexdigest()

    (package / "training.py").write_text("modified", encoding="utf-8")
    changed, modified_files = provenance.code_provenance()
    assert changed != original
    assert modified_files["policy_relative_lora.py"] == files["policy_relative_lora.py"]
    assert (
        modified_files["policy_relative/training.py"]
        != files["policy_relative/training.py"]
    )

    (root / "intents/generation/backend.py").write_text("modified", encoding="utf-8")
    changed_again, shared_files = provenance.code_provenance()
    assert changed_again != changed
    assert (
        shared_files["intents/generation/backend.py"]
        != files["intents/generation/backend.py"]
    )

    (root / "intents/provenance.py").write_text("modified", encoding="utf-8")
    changed_helper, helper_files = provenance.code_provenance()
    assert changed_helper != changed_again
    assert helper_files["intents/provenance.py"] != files["intents/provenance.py"]

    normalizer = root / "intents/representation/normalization.py"
    normalizer.write_text("modified", encoding="utf-8")
    changed_normalizer, normalizer_files = provenance.code_provenance()
    assert changed_normalizer != changed_helper
    assert (
        normalizer_files["intents/representation/normalization.py"]
        != files["intents/representation/normalization.py"]
    )
    assert "intent_fence_audit.py" not in normalizer_files


def test_existing_module_cli_remains_available():
    result = subprocess.run(
        [sys.executable, "-m", "carl.icl.policy_relative_lora", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{inspect,resource-check,benchmark,train,evaluate}" in result.stdout
    assert "--requested-scale" in result.stdout
