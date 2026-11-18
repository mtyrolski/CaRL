"""Build Plotly graph_objects diagnostics from saved Luna ICL runs, fully offline."""
import argparse
from typing import Final

from carl.icl.evaluation.data import EvaluationConfig, load_evaluation_config
from carl.icl.evaluation.report import evaluate

DEFAULT_CONFIG: Final[str] = "configs/icl/evaluation/mock.json"


def main() -> int:
    """Load evaluation settings with optional input/output overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runs", nargs="+", help="Condition or matrix directories; overrides run_dirs"
    )
    parser.add_argument("--output", help="Override the output directory")
    parser.add_argument(
        "--formats",
        nargs="*",
        choices=("png", "svg", "pdf"),
        help="Optional static exports via Kaleido",
    )
    args = parser.parse_args()
    settings = load_evaluation_config(args.config).model_dump()
    for argument, key in (
        ("runs", "run_dirs"),
        ("output", "output_dir"),
        ("formats", "static_formats"),
    ):
        value = getattr(args, argument)
        if value is not None:
            settings[key] = value
    print(evaluate(EvaluationConfig.model_validate(settings)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Evaluation input/export error: {error}") from None
