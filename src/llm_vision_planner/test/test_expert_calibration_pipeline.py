#!/usr/bin/env python3
"""Exercise both expert CSV schemas through the shared offline CLI pipeline."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from conformal_semantic_theta_dataset import SOURCE_FIELDNAMES
from semantic_theta import DEFAULT_SEMANTIC_POLICY, DEFAULT_WORKSPACE, plan_semantic_theta


class ExpertCalibrationPipelineTest(unittest.TestCase):
    def test_expert_selection_preserves_qp_clock_scores_and_plot_outputs(self):
        start = {"x": 0.0, "y": 0.0, "z": -0.25}
        goal = {"x": 1.0, "y": 0.0, "z": -0.25}
        expert_path = plan_semantic_theta(start, goal, [], DEFAULT_WORKSPACE)
        llm_paths = [expert_path, [start, {"x": 0.5, "y": 0.2, "z": -0.25}, goal]]
        results = {}
        with tempfile.TemporaryDirectory(prefix="expert-calibration-test-") as temporary:
            directory = Path(temporary)

            def run(script, *args, check=True):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / script), *map(str, args)],
                    cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"},
                    capture_output=True, text=True, timeout=90,
                )
                if check:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            for expert in ("rrt", "semantic_theta"):
                with self.subTest(expert=expert):
                    source = directory / f"{expert}_pairs.csv"
                    qp = directory / f"{expert}_qp.csv"
                    scored = directory / f"{expert}_scores.csv"
                    report = directory / f"{expert}.json"
                    plot = directory / f"{expert}.png"
                    fields = [name.replace("semantic_theta", expert) for name in SOURCE_FIELDNAMES]
                    with source.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        for index, llm_path in enumerate(llm_paths):
                            writer.writerow({
                                "sample_id": index,
                                "start": json.dumps(start), "goal": json.dumps(goal),
                                "workspace": json.dumps(DEFAULT_WORKSPACE), "obstacles": "[]",
                                "semantic_policy": json.dumps(DEFAULT_SEMANTIC_POLICY),
                                f"{expert}_waypoints": json.dumps(expert_path),
                                f"{expert}_verified_waypoints": json.dumps(expert_path),
                                "llm_waypoints": json.dumps(llm_path),
                                "llm_verified_waypoints": json.dumps(llm_path),
                            })
                    # No flag for RRT: verify backward compatibility of existing commands.
                    selection = ["--expert", expert] if expert == "semantic_theta" else []
                    if selection:
                        rejected = run("generate_qp_calibration_dataset.py", "--input", source,
                                       "--output", qp, "--samples", 2, check=False)
                        self.assertNotEqual(rejected.returncode, 0)
                        self.assertIn("rrt_verified_waypoints", rejected.stderr)
                    run("generate_qp_calibration_dataset.py", *selection, "--input", source,
                        "--output", qp, "--samples", 2, "--delta-u", 0.5, "--delta-x", 0.5)
                    run("generate_position_score_calibration.py", *selection, "--input", qp,
                        "--output", scored, "--samples", 2, "--delta-p", 0.5, "--delta-w", 0.5)
                    # Calibrate on the identity pair; the off-path second pair must fail.
                    run("dconformal_contraction_verify.py", *selection, "--calibration-csv", scored,
                        "--calibration-samples", 1, "--sample-id", 1,
                        "--delta-p", 0.5, "--delta-w", 0.5,
                        "--output-png", plot, "--report-json", report)
                    with scored.open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    details = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(details["expert"], expert)
                    self.assertFalse(details["score_accepted"])
                    self.assertGreater(details["s_p"], details["q_p"])
                    self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    for row in rows:
                        reference = json.loads(row[f"{expert}_trajectory"])["samples"]
                        llm = json.loads(row["llm_trajectory"])["samples"]
                        self.assertEqual([s["t"] for s in reference], [s["t"] for s in llm])
                        self.assertEqual(json.loads(row["semantic_policy"]), DEFAULT_SEMANTIC_POLICY)
                    self.assertLess(float(rows[0]["s_p"]), 1e-5)
                    self.assertGreater(float(rows[1]["s_p"]), 0.01)
                    results[expert] = rows

            for rrt, semantic in zip(results["rrt"], results["semantic_theta"]):
                self.assertEqual(rrt["rrt_trajectory"], semantic["semantic_theta_trajectory"])
                for field in ("llm_trajectory", "s_u", "s_x", "q_u", "q_x", "s_p", "s_w", "q_p", "q_w"):
                    self.assertEqual(rrt[field], semantic[field], field)


if __name__ == "__main__":
    unittest.main()
