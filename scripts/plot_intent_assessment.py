#!/usr/bin/env python3
"""Plot measured held-out validity and action sequences found by bounded search."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main():
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    summaries = data["results"]
    labels = {
        "base": "Base model",
        "adapted": "LoRA adapter",
        "in_context": "In-context",
        "local_rule": "Local rule",
    }
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(6.7, 3.9), layout="constrained")
    x = np.arange(len(summaries))
    width = 0.34
    metrics = [
        (
            "board_mean_valid_fraction",
            "valid_fraction_95_board_bootstrap",
            "valid_nontrivial",
            "Valid intent",
            "#496f8c",
        ),
        (
            "board_mean_witness_fraction",
            "witness_fraction_95_board_bootstrap",
            "witness_found",
            "Action sequence found",
            "#bd7756",
        ),
    ]
    for j, (field, ci_field, numerator, label, color) in enumerate(metrics):
        for i, summary in enumerate(summaries):
            value = summary[field]
            xpos = x[i] + (j - 0.5) * width
            if value is None:
                ax.text(
                    xpos,
                    0.025,
                    "not run",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                    color="0.3",
                )
                continue
            ax.bar(
                xpos,
                value,
                width=width * 0.93,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ci = summary[ci_field]
            if ci is not None:
                # A percentile interval can exclude the point estimate; draw
                # the actual endpoints rather than forcing symmetric errors.
                ax.vlines(xpos, ci[0], ci[1], color="black", linewidth=0.85)
                ax.hlines(
                    [ci[0], ci[1]],
                    xpos - 0.045,
                    xpos + 0.045,
                    color="black",
                    linewidth=0.85,
                )
            label_height = max(value, ci[1] if ci is not None else value) + 0.025
            ax.text(
                xpos,
                min(1.09, label_height),
                f"{summary[numerator]}/{summary['outputs']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Mean fraction per held-out board")
    if args.title:
        ax.set_title(args.title)
    ax.set_xticks(x, [labels[s["method"]] for s in summaries])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".50", ".75", "1"])
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for _, _, _, label, color in metrics],
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=9,
    )
    comparison_note = ("Intents differ in difficulty; the local rule searches legal moves."
                       if any(row["method"] == "local_rule" for row in summaries) else
                       "Intents differ in difficulty; found paths satisfy their proposed intents.")
    fig.text(
        0.5,
        -0.045,
        "Counts: proposals. Whiskers: 95% board bootstrap.\n"
        f"{comparison_note}\nNo controller or full-puzzle success is measured.",
        ha="center",
        fontsize=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.output.with_suffix("." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
