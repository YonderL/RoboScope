"""Rebuild the official per-task table from portable episode records (CPU only)."""

import argparse
import csv
import json
from pathlib import Path

from roboscope.reporting.official import build_summary


def export(results, output):
    summary = build_summary(results)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "per_task.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["per_task"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["per_task"])
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/official_spatial"))
    args = parser.parse_args()
    export(args.results, args.output)
