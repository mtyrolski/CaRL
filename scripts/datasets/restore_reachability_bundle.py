"""Restore a frozen reachability dataset from its verified private source bundle.

No network, source shard or model is needed. Do not use bundles from untrusted
sources: they contain private evaluation witnesses and archived source code.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from carl.icl.reachability_data import _build, _unbundle, verify_dataset

ROOT = Path(__file__).resolve().parents[2]


def restore(bundle: Path, output: Path, *, expected_sha256: str, seed: int, count: int) -> dict:
    """Recreate frozen artifacts in a new directory, without overwriting."""
    data = bundle.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Private source bundle checksum mismatch")
    files = _unbundle(data)
    runtime = json.loads(files["runtime/metadata.json"])
    if any(not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
           for name, expected in runtime["code_sha256"].items()):
        raise ValueError("Runtime source or lockfile differs from archived bundle")
    if runtime["python"] != sys.version.split()[0] or runtime["packages"]["numpy"] != np.__version__:
        raise ValueError("Python or NumPy differs from archived bundle")
    artifacts = _build(files, seed=seed, n_main=count)
    if artifacts["private/source_bundle.tar.gz"] != data:
        raise ValueError("Archived source bundle is not canonical")
    if output.exists():
        raise ValueError("Output already exists; choose a new destination")
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("private/"):
            destination.parent.chmod(0o700)
        destination.write_bytes(content)
        if name.startswith("private/"):
            destination.chmod(0o600)
    return verify_dataset(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Expected SHA-256 from the frozen manifest")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    result = restore(args.bundle, args.output, expected_sha256=args.sha256, seed=args.seed, count=args.count)
    print(json.dumps({"status": "restored_and_verified", "output": str(args.output),
                      "schema": result["schema"], "core_pairs": result["core_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
