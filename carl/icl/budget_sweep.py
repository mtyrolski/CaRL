"""Validated, resumable solver runs at independently configured action budgets."""
import hashlib
import json
import re
import sys
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Final

from pydantic import Field, model_validator

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT, ExperimentConfig, Settings
from carl.icl.datasets import load_data
from carl.icl.experiment import (
    calibration_path,
    components_for_run,
    credentials,
    matrix_configs,
    run_condition,
)
from carl.icl.reporting import report
from carl.icl.types import JSONObject

CALIBRATED_METHODS: Final[frozenset[str]] = frozenset({
    "relative", "relative_u", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset",
})
MAIN_COMPARISON: Final[frozenset[str]] = frozenset({"relative_u", "absolute_d"})
OPTIONAL_METHODS: Final[frozenset[str]] = frozenset({"ratio_d_over_ac", "normalized_offset", "fixed"})
LEGACY_COMPARISON: Final[frozenset[str]] = frozenset({"relative", "fixed"})

_CODE_FILES: Final[tuple[str, ...]] = (
    "carl/icl/api.py",
    "carl/icl/boards.py",
    "carl/icl/calibration.py",
    "carl/icl/config.py",
    "carl/icl/datasets.py",
    "carl/icl/experiment.py",
    "carl/icl/experiment_api.py",
    "carl/icl/luna.py",
    "carl/icl/prompt_defaults.py",
    "carl/icl/responses.py",
    "carl/icl/reporting.py",
    "carl/icl/search.py",
    "carl/icl/types.py",
    "carl/icl/budget_sweep.py",
    "carl/environment/env.py",
    "carl/environment/tokenizer.py",
    "carl/environment/utilis.py",
    "carl/environment/sokoban/core.py",
    "carl/environment/sokoban/env.py",
    "carl/environment/sokoban/tokenizer.py",
)
CODE_FILES: Final[tuple[str, ...]] = tuple(dict.fromkeys((
    *_CODE_FILES,
    *(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "carl/icl").rglob("*.py"))),
)))


class BudgetSweepConfig(Settings):
    """A matrix of matched methods evaluated at explicit action budgets."""

    matrix: str
    budgets: list[int] = Field(min_length=2)
    output_dir: str

    @model_validator(mode="after")
    def validate_budgets(self) -> "BudgetSweepConfig":
        """Make budget order, uniqueness and bounds explicit."""
        if any(budget < 1 for budget in self.budgets) or self.budgets != sorted(
            set(self.budgets)
        ):
            raise ValueError(
                "Budgets must be positive, distinct and strictly increasing"
            )
        return self


def load_sweep(path: str | Path) -> tuple[BudgetSweepConfig, list[ExperimentConfig]]:
    """Resolve the matrix relative to the sweep specification."""
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    spec = BudgetSweepConfig.model_validate_json(path.read_text(encoding="utf-8"))
    matrix = Path(spec.matrix).expanduser()
    configs = matrix_configs(matrix if matrix.is_absolute() else path.parent / matrix)
    if not configs:
        raise ValueError("The sweep matrix must contain conditions")
    for family in {config.family for config in configs}:
        group = [config for config in configs if config.family == family]
        methods = [config.method for config in group]
        distinct = set(methods)
        if len(distinct) != len(methods):
            raise ValueError(f"Family {family} has multiple conditions for the same method")
        if distinct != LEGACY_COMPARISON and not (
            MAIN_COMPARISON <= distinct <= MAIN_COMPARISON | OPTIONAL_METHODS
        ):
            raise ValueError(
                f"Family {family} must contain relative_u and absolute_d, optionally ratio_d_over_ac, "
                "normalized_offset and fixed; the legacy relative/fixed pair is also supported"
            )
        if any(config.easy for config in group):
            raise ValueError("A budget sweep must use full initial states (easy=false)")
    for config in configs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config.name):
            raise ValueError(
                "Condition names must contain only letters, numbers, underscores or hyphens"
            )
        if not config.repeat_seeds or len(set(config.repeat_seeds)) != len(
            config.repeat_seeds
        ):
            raise ValueError("Repeat seeds must be nonempty and unique")
    return spec, configs


def planned_configs(
    spec: BudgetSweepConfig, configs: list[ExperimentConfig]
) -> list[tuple[int, ExperimentConfig]]:
    """Keep all non-budget settings fixed across the entire sweep."""
    return [
        (
            budget,
            ExperimentConfig.model_validate(
                {**config.model_dump(), "action_budget": budget}
            ),
        )
        for budget in spec.budgets
        for config in configs
    ]


def describe_plan(
    spec: BudgetSweepConfig, configs: list[ExperimentConfig]
) -> JSONObject:
    """Report planned solver trials, without making provider calls."""
    return {
        "budgets": spec.budgets,
        "conditions": [
            {
                "name": config.name,
                "method": config.method,
                "family": config.family,
                "instances": config.instances,
                "seeds": config.repeat_seeds,
                "scales": config.scales,
                "fixed_cap": config.fixed_cap,
            }
            for config in configs
        ],
        "solver_runs": sum(
            config.instances * len(config.repeat_seeds) for config in configs
        )
        * len(spec.budgets),
        "output_dir": spec.output_dir,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    spec: BudgetSweepConfig, configs: list[ExperimentConfig], provider: str
) -> JSONObject:
    """Bind resume to exact data, calibration, code, dependencies and run settings."""
    inputs: dict[str, str] = {}
    for config in configs:
        training, instances = load_data(config)
        components = components_for_run(
            config, provider, training, from_rows(instances[0]["state"])
        )
        data = config.path("data_dir")
        paths = [data / "training.json", data / config.family / f"{config.split}.json"]
        if config.value_examples == "trajectory":
            paths.append(data / "value_examples.json")
        if config.method in CALIBRATED_METHODS:
            paths.append(calibration_path(config, components))
        for path in paths:
            inputs[str(path.resolve())] = _sha(path)
    return {
        "version": 1,
        "provider": provider,
        "budgets": spec.budgets,
        "conditions": [config.model_dump(mode="json") for config in configs],
        "inputs_sha256": inputs,
        "code_sha256": {path: _sha(ROOT / path) for path in CODE_FILES},
        "python_version": sys.version,
        "package_versions": {
            name: version(name)
            for name in ("openai", "numpy", "pydantic", "tiktoken", "scipy", "torch")
        },
    }


@contextmanager
def _sweep_lock(output: Path) -> Iterator[None]:
    """Prevent two processes from mutating the same sweep at the same time."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".sweep.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                f"Another process is running this sweep: {output}"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_sweep(
    spec: BudgetSweepConfig,
    configs: list[ExperimentConfig],
    provider: str,
    output: Path,
    *,
    resume: bool,
) -> None:
    """Run or resume each budget with preflight and per-trial atomic records."""
    output = output.expanduser().resolve()
    with _sweep_lock(output):
        _run_locked(spec, configs, provider, output, resume=resume)


def _run_locked(
    spec: BudgetSweepConfig,
    configs: list[ExperimentConfig],
    provider: str,
    output: Path,
    *,
    resume: bool,
) -> None:
    """Require an immutable manifest before creating or continuing trials."""
    manifest_path = output / "sweep_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise ValueError(f"Cannot resume without {manifest_path}")
    elif any(path.name != ".sweep.lock" for path in output.iterdir()):
        raise ValueError(
            f"Output is not empty: {output}; use --resume or another directory"
        )
    manifest = _manifest(spec, configs, provider)
    if resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != saved:
            raise ValueError(
                "Cannot resume: provider, settings, data, calibration, code or dependencies changed"
            )
    else:
        if all(config.roles is None for config in configs):
            credentials(provider)
        write_json(manifest_path, manifest)
        write_json(
            output / "sweep_plan.json",
            {**describe_plan(spec, configs), "output_dir": str(output)},
        )
    for budget in spec.budgets:
        rows: list[JSONObject] = []
        for _, config in (
            item for item in planned_configs(spec, configs) if item[0] == budget
        ):
            directory = output / f"budget_{budget:06d}" / config.name
            rows.extend(
                run_condition(
                    config, provider, directory, resume=True, stop_on_api_error=True
                )
            )
            report(rows, directory.parent)
    print(f"Completed sweep: {output}", flush=True)
