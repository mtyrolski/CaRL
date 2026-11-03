"""Export a parameter grid for legacy trajectory-target experiments; does not train."""
import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from carl.icl.api import write_json


@dataclass(frozen=True)
class TargetConfig:
    mode: str
    k: int | None = None
    offsets: tuple[int, ...] = ()
    radius: int = 0
    distance_interval: tuple[int, ...] | None = None


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result:
        raise ValueError("Expected comma-separated integers")
    return result


def _parse_optional_int_csv(value: str) -> tuple[int, ...] | None:
    return None if value.lower().strip() == "none" else _parse_int_csv(value)


def _build_grid(modes, k_values, offsets_sets, radii, distance_intervals) -> list[TargetConfig]:
    if any(k <= 0 for k in k_values) or any(r < 0 for r in radii):
        raise ValueError("Horizons must be positive and radii nonnegative")
    configs = []
    for mode in modes:
        if mode == "sliding_window":
            configs.extend(
                TargetConfig(mode, distance_interval=distances) for distances in distance_intervals
            )
        elif mode == "k_offsets":
            configs.extend(
                TargetConfig(mode, k=k, offsets=offsets, radius=radius)
                for k, offsets, radius in product(k_values, offsets_sets, radii)
            )
        elif mode == "hierarchical_raw":
            configs.extend(TargetConfig(mode, k=k) for k in k_values)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=_parse_int_csv, default=(8, 4, 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = _build_grid(
        ("sliding_window", "k_offsets", "hierarchical_raw"),
        args.k,
        ((-1, 0, 1),),
        (0, 1),
        (None, args.k),
    )
    write_json(args.output, [asdict(config) for config in configs])


if __name__ == "__main__":
    main()
