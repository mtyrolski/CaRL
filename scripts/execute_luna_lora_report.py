#!/usr/bin/env python3
"""Execute the offline Luna/LoRA notebook without model or API calls."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute(source: Path, output: Path, run_dir: Path, root: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    previous = os.environ.get("CARL_LUNA_LORA_RUN")
    os.environ["CARL_LUNA_LORA_RUN"] = str(run_dir.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                resources={"metadata": {"path": str(root.resolve())}})
        client.execute()
    finally:
        if previous is None:
            os.environ.pop("CARL_LUNA_LORA_RUN", None)
        else:
            os.environ["CARL_LUNA_LORA_RUN"] = previous
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            nbformat.write(notebook, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.output, args.run_dir, Path(__file__).resolve().parents[1])
    print(f"Executed offline report: {args.output}")


if __name__ == "__main__":
    main()
