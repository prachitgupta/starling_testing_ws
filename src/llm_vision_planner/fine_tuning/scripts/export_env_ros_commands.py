#!/usr/bin/env python3
"""Export calibration environments with ready-to-run simulator obstacle commands."""

import argparse
import csv
import json
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv",
    )
    parser.add_argument(
        "--output",
        default="fine_tuning/datasets/env_ros_commands.csv",
    )
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    with source.open(newline="") as input_file, target.open("w", newline="") as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(
            output_file,
            fieldnames=["sample_id", "environment", "ros2_pub_command", "raw_rrt", "raw_llm", "s_u", "s_x"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            environment = {
                key: json.loads(row[key])
                for key in ("start", "goal", "workspace", "obstacles")
            }
            obstacle_message = json.dumps(
                {"obstacles": environment["obstacles"], "timestamp": 0.0}, separators=(",", ":")
            )
            yaml_argument = json.dumps({"data": obstacle_message}, separators=(",", ":"))
            command = (
                "ros2 topic pub -r 2 /llm_vision/sim_obstacles std_msgs/msg/String "
                f"{shlex.quote(yaml_argument)}"
            )
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "environment": json.dumps(environment, separators=(",", ":")),
                    "ros2_pub_command": command,
                    "raw_rrt": row["rrt_waypoints"],
                    "raw_llm": row["llm_waypoints"],
                    "s_u": row["s_u"],
                    "s_x": row["s_x"],
                }
            )


if __name__ == "__main__":
    main()
