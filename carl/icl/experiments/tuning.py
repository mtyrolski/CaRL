"""Validation-only scale-grid selection for legacy experiment configurations."""
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.config import ExperimentConfig
from carl.icl.reporting import report
from carl.icl.types import JSONObject


def tune(
    config: ExperimentConfig, provider: str, directory: Path, *,
    run_condition: Callable[[ExperimentConfig, str, Path], list[JSONObject]],
) -> None:
    """Equal-size grids, validation-only selection, frozen scales for transfer."""
    if config.easy:
        raise ValueError("Do not tune scales on one-move interface fixtures")
    grids: dict[str, Sequence[tuple[float, float]]] = {
        "relative": config.tuning.relative_scales,
        "fixed": config.tuning.fixed_scales,
    }
    if (
        max(max(pair) for pair in config.tuning.fixed_scales)
        > config.calibration.action_cap
    ):
        raise ValueError(
            "Tuning horizons exceed the controller cap; adjust the tuning grid before inference"
        )
    all_rows, selections = [], {}
    for method, grid in grids.items():
        scores = []
        for index, scales in enumerate(grid):
            candidate = ExperimentConfig.model_validate(
                dict(
                    config.model_dump(),
                    name=f"tune_{method}_{index}",
                    method=method,
                    scales=scales,
                    split="validation",
                    family="real_12x12x4",
                    instances=config.validation_boards,
                )
            )
            rows = run_condition(candidate, provider, directory / candidate.name)
            all_rows.extend(rows)
            if any(row["finish_reason"] == "api_error" for row in rows):
                report(all_rows, directory)
                raise ValueError("Cannot select scales from runs with API errors")
            score = (
                sum(row["solved"] for row in rows),
                -sum(row["attempted_actions"] for row in rows),
                -index,
            )
            scores.append((score, scales))
            report(all_rows, directory)
        chosen = max(scores)[1]
        selections[method] = dict(
            config.model_dump(),
            name=f"selected_{method}",
            method=method,
            scales=chosen,
            tuning_status="pilot_unvalidated"
            if provider == "mock"
            else "validation_selected",
            family="real_12x12x4",
            split="test",
            easy=False,
        )
        write_json(directory / f"selected_{method}.json", selections[method])
        write_json(
            directory / f"selected_{method}_32.json",
            dict(
                selections[method],
                name=f"selected_{method}_32",
                family="generated_32x32x10",
            ),
        )
    write_json(
        directory / "matrix_selected.json",
        {
            "configs": [
                "selected_relative.json",
                "selected_fixed.json",
                "selected_relative_32.json",
                "selected_fixed_32.json",
            ]
        },
    )
    write_json(
        directory / "selection.json",
        {
            "provider": provider,
            "criterion": "validation solved count, then attempted actions, then grid order",
            "grid": grids,
            "selections": selections,
            "data_sha256": hashlib.sha256(
                (config.path("data_dir") / "training.json").read_bytes()
            ).hexdigest(),
        },
    )
