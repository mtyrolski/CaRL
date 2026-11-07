"""Restore locally archived reachability scripts without overwriting files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "analysis/icl_reachability/cleanup_backup/legacy_sources"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--list", action="store_true", help="List archived paths without restoring")
    args = parser.parse_args()
    archive = args.archive.resolve()
    records = json.loads((archive / "manifest.json").read_text())["files"]
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid archive path: {relative}")
        source = archive / relative
        destination = ROOT / relative
        if not source.is_file() or digest(source) != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {relative}")
        if destination.exists() and (not destination.is_file() or digest(destination) != record["sha256"]):
            raise FileExistsError(f"Refusing to overwrite: {relative}")
    if args.list:
        for record in records:
            print(record["path"])
        return
    restored = 0
    for record in records:
        source = archive / record["path"]
        destination = ROOT / record["path"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            restored += 1
    print(f"Restored {restored} files; existing matching files were left untouched.")


if __name__ == "__main__":
    main()
