"""Export campaign data integrity and witness-length summaries without solutions."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "analysis/luna_icl/campaign_100"


def main() -> int:
    """Write human- and machine-readable split statistics."""
    provenance = json.loads((DIRECTORY / "data_integrity.json").read_text(encoding="utf-8"))
    summary = provenance["counts_and_witness_statistics"]
    rows = []
    for name, record in summary["groups"].items():
        split, family = name.split("/")
        rows.append({"split": split, "family": family, "count": record["count"],
                     "unique_layouts": record["unique_layouts"], **record["known_witness_actions"]})
    with (DIRECTORY / "dataset_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DIRECTORY / "split_intersections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["left", "right", "canonical_layout_intersection"])
        writer.writeheader()
        for pair, count in summary["intersections"].items():
            left, right = pair.split(" | ")
            writer.writerow({"left": left, "right": right, "canonical_layout_intersection": count})
    lines = ["# Integralność i długości świadków", "",
             f"Przeindeksowane shardy: {provenance['shards_indexed']}; layouty powtórzone "
             f"między shardami: {provenance['cross_shard_duplicate_layouts']}.", "",
             "| Split | Rodzina | Liczba | Unikalne layouty | Min | Q25 | Mediana | Średnia | Q75 | Max |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['family']} | {row['count']} | {row['unique_layouts']} | "
                     f"{row['min']} | {row['q25']:.2f} | {row['median']:.2f} | {row['mean']:.2f} | "
                     f"{row['q75']:.2f} | {row['max']} |")
    lines.extend(["", f"Przecięcia: {len(summary['intersections'])} par; "
                  f"niezerowych: {sum(value != 0 for value in summary['intersections'].values())}. "
                  "Wszystkie pary są w `split_intersections.csv`.", "",
                  "Długość to liczba akcji znanego świadka rozwiązania; nie jest minimalną "
                  "długością rozwiązania ani dowodem optymalności."])
    witness_report = DIRECTORY / "witness_verification.json"
    if witness_report.exists():
        verified = json.loads(witness_report.read_text(encoding="utf-8"))
        lines.extend(["", f"Odtworzono {verified['witnesses_verified']} świadków "
                      f"i sprawdzono {verified['primitive_actions_verified']} akcji; "
                      "wszystkie stany końcowe spełniły `is_solved`."])
    reproducibility = DIRECTORY / "reproducibility.json"
    if reproducibility.exists():
        repeated = json.loads(reproducibility.read_text(encoding="utf-8"))
        lines.extend(["", "Powtórne generowanie: "
            f"holdout {repeated['synthetic_holdout_exact_matches']}/{repeated['synthetic_holdout_total']}, "
            f"pilot {repeated['synthetic_pilot_state_layout_witness_matches']}/{repeated['synthetic_pilot_total']}, "
            f"walidacja skal {repeated['synthetic_scale_exact_matches']}/{repeated['synthetic_scale_total']}, "
            f"walidacja komponentów {repeated['synthetic_component_state_layout_matches']}/{repeated['synthetic_component_total']}."])
    (DIRECTORY / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote aggregate data report and tables to {DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
