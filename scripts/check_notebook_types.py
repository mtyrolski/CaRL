"""Type-check Python notebook cells together and report notebook cell locations."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_notebook(notebook: Path, destination: Path) -> dict[int, tuple[int, int]]:
    """Write code cells in execution order and map generated lines to cell lines."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    lines: list[str] = []
    locations: dict[int, tuple[int, int]] = {}
    for cell_index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines.append(f"# Notebook cell {cell_index}\n")
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        for cell_line, line in enumerate(text.splitlines(), start=1):
            lines.append(line + "\n")
            locations[len(lines)] = (cell_index, cell_line)
        lines.append("\n")
    destination.write_text("".join(lines), encoding="utf-8")
    return locations


def main() -> int:
    """Run the installed Pyright without creating files in the checkout."""
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        print("No notebooks to type-check.")
        return 0
    with tempfile.TemporaryDirectory(prefix="carl-notebook-types-") as temporary:
        locations = {}
        for notebook in notebooks:
            extracted = Path(temporary) / f"{notebook.stem}.py"
            locations[str(extracted)] = (notebook, extract_notebook(notebook, extracted))
        process = subprocess.run(
            [sys.executable, "-m", "pyright", "--outputjson", *locations],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            report = json.loads(process.stdout)
        except json.JSONDecodeError:
            print(process.stderr or process.stdout, file=sys.stderr)
            return process.returncode or 1
        for diagnostic in report["generalDiagnostics"]:
            path, cell_lines = locations[diagnostic["file"]]
            line = diagnostic["range"]["start"]["line"] + 1
            cell, offset = cell_lines.get(line, (-1, line))
            column = diagnostic["range"]["start"]["character"] + 1
            rule = diagnostic.get("rule", "diagnostic")
            print(f"{path.relative_to(ROOT)}:cell {cell}:line {offset}:{column}: "
                  f"{diagnostic['severity']} [{rule}] {diagnostic['message']}")
        summary = report["summary"]
        print(f"Notebook Pyright: {summary['errorCount']} errors, "
              f"{summary['warningCount']} warnings across {len(notebooks)} notebooks")
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
