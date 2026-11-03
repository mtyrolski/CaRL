"""Compare independently regenerated campaign data without displaying solutions."""
import hashlib
import json
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/icl"
REPORT = ROOT / "analysis/luna_icl/campaign_100/reproducibility.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_files(source: Path, regenerated: Path, start: int, count: int) -> int:
    """Compare byte-for-byte private staging artifacts without exposing their contents."""
    matches = 0
    for seed in range(start, start + count):
        first = (source / f"{seed}.json").read_bytes()
        second = (regenerated / f"{seed}.json").read_bytes()
        matches += hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    return matches


def main() -> int:
    holdout = exact_files(DATA / ".private_campaign_holdout_staging",
                          Path("/tmp/carl_campaign_synthetic_repro_20261102"), 2026120200, 500)
    scale = exact_files(DATA / ".private_campaign_scale_staging",
                        Path("/tmp/carl_campaign_scale_repro_20261102"), 2026111500, 4)
    pilot = 0
    for index in range(50):
        original = read(DATA / "campaign_100_pilot_staging" / f"{index:03d}.json")
        repeated = read(Path("/tmp/carl_campaign_pilot_repro_20261102") / f"{2026110200 + index}.json")
        state = from_rows(original["state"])
        pilot += (state_hash(state) == repeated["state_hash"]
                  and canonical_layout_id(state) == repeated["layout_id"]
                  and original["actions"] == repeated["actions"])
    legacy = read(DATA / "manifest.json")
    seeds = {record["id"]: record["seed"] for record in legacy["generation"]}
    records = read(DATA / "generated_32x32x10/validation.json")["instances"]
    component = 0
    for record in records:
        repeated = read(Path("/tmp/carl_campaign_component_repro_20261102") / f"{seeds[record['id']]}.json")
        state = from_rows(record["state"])
        component += (state_hash(state) == repeated["state_hash"]
                      and canonical_layout_id(state) == repeated["layout_id"])
    result = {"synthetic_holdout_exact_matches": holdout, "synthetic_holdout_total": 500,
              "synthetic_scale_exact_matches": scale, "synthetic_scale_total": 4,
              "synthetic_pilot_state_layout_witness_matches": pilot, "synthetic_pilot_total": 50,
              "synthetic_component_state_layout_matches": component, "synthetic_component_total": len(records)}
    if (holdout, scale, pilot, component) != (500, 4, 50, len(records)):
        raise ValueError("Independently regenerated synthetic data differ from the frozen originals")
    write_json(REPORT, result)
    print("Reproduction matches: holdout 500/500, scale 4/4, pilot 50/50, component validation "
          f"{component}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
