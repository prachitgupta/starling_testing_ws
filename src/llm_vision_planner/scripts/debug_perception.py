#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class DebugPerception(Node):
    def __init__(self):
        super().__init__("debug_perception")
        self.declare_parameter("obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("show_window", True)
        self.declare_parameter("output_png", "")
        self.payload = None
        self.figure, self.axis = plt.subplots(figsize=(8, 7))
        if bool(self.get_parameter("show_window").value):
            plt.ion()
            self.figure.show()

        topic = str(self.get_parameter("obstacle_topic").value)
        self.create_subscription(String, topic, self.obstacle_callback, 10)
        self.create_timer(0.25, self.render)
        self.get_logger().info(f"Plotting semantic obstacles from {topic}")

    def obstacle_callback(self, msg):
        try:
            self.payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid obstacle JSON: {exc}")

    def render(self):
        if self.payload is None:
            return

        pose = self.payload.get("pose") or [0.0, 0.0, 0.0, 0.0]
        drone_x, drone_y = float(pose[0]), float(pose[1])
        obstacles = self.payload.get("obstacles", [])

        self.axis.clear()
        self.axis.scatter(drone_x, drone_y, s=120, marker="^", color="tab:blue", label="Drone", zorder=3)
        self.axis.annotate("Drone", (drone_x, drone_y), xytext=(6, 6), textcoords="offset points")

        x_values = [drone_x]
        y_values = [drone_y]
        for obstacle in obstacles:
            minimum = obstacle.get("min_corner", [0.0, 0.0, 0.0])
            maximum = obstacle.get("max_corner", [0.0, 0.0, 0.0])
            min_x, min_y = float(minimum[0]), float(minimum[1])
            max_x, max_y = float(maximum[0]), float(maximum[1])
            width, depth = max_x - min_x, max_y - min_y
            center_x, center_y = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
            distance = float(obstacle.get("distance_m", math.hypot(center_x - drone_x, center_y - drone_y)))
            label = str(obstacle.get("label") or obstacle.get("shape") or "object")

            self.axis.add_patch(
                Rectangle((min_x, min_y), width, depth, facecolor="tab:red", edgecolor="darkred", alpha=0.3)
            )
            self.axis.plot([drone_x, center_x], [drone_y, center_y], "--", color="0.45", linewidth=1)
            self.axis.annotate(
                f"{label}\n{distance:.2f} m\n{width:.2f} × {depth:.2f} m",
                (center_x, center_y),
                ha="center",
                va="center",
                fontsize=9,
            )
            x_values.extend([min_x, max_x])
            y_values.extend([min_y, max_y])

        margin = 0.75
        self.axis.set_xlim(min(x_values) - margin, max(x_values) + margin)
        self.axis.set_ylim(min(y_values) - margin, max(y_values) + margin)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, alpha=0.3)
        self.axis.set_xlabel("NED x / North (m)")
        self.axis.set_ylabel("NED y / East (m)")
        status = self.payload.get("status", {})
        self.axis.set_title(
            f"Semantic obstacle debug — {len(obstacles)} object(s) — healthy={self.payload.get('healthy', False)}\n"
            f"detector={status.get('detector', 'unknown')}, depth={status.get('point_cloud', 'unknown')}, "
            f"pose={status.get('pose', 'unknown')}"
        )
        if not obstacles:
            self.axis.text(0.5, 0.5, "No semantic obstacles detected", transform=self.axis.transAxes, ha="center")

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        output_png = str(self.get_parameter("output_png").value).strip()
        if output_png:
            output = Path(output_png)
            output.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(output, dpi=140)


def main():
    rclpy.init()
    node = DebugPerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
