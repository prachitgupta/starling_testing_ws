#!/usr/bin/env python3
"""Generate new 2,000-row trajectory datasets without replacing 200-row files."""

import argparse
import csv
from pathlib import Path

from generate_xu_comparison_datasets import DEFAULT_INPUT, build_rows, write_rows


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "datasets"
OUTPUTS = {
    "different_clocks": "calibration_min_snap_different_clocks_2000.csv",
    "shared_clock": "calibration_min_snap_shared_clock_2000.csv",
    "frenet": "calibration_min_snap_frenet_2000.csv",
    "guided_shared": "calibration_min_snap_guided_fit_shared_2000.csv",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))[: args.samples]
    if len(source_rows) != args.samples:
        raise RuntimeError(f"Input contains {len(source_rows)} rows; requested {args.samples}.")
    for method, filename in OUTPUTS.items():
        output = args.output_dir / filename
        if output.exists():
            raise FileExistsError(f"Refusing to replace existing calibration file: {output}")
        rows = build_rows(source_rows, method, args.dt, args.delta_u, args.delta_x)
        write_rows(output, rows)
        print(f"Wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
