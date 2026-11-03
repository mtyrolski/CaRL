"""Export interactive plots from a saved Luna component study without API calls."""
import argparse
from pathlib import Path

from carl.icl.config import ROOT
from carl.icl.evaluation.component_study import export


def main() -> int:
    """Export four Plotly HTML files and CSV summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="analysis/luna_icl/component_study_100")
    args = parser.parse_args()
    path = Path(args.directory).expanduser()
    print(f"Plots and tables: {export(path if path.is_absolute() else ROOT / path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
