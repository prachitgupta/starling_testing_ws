#!/usr/bin/env python3
import math

import px4_ros2
import rclpy
from rclpy.node import Node


class GotoMode(px4_ros2.components.ModeBase):
    def __init__(self, node, x, y, z, speed, heading):
        super().__init__(node=node, mode_name="VOXL Goto Smoke Test")
        self.goto = px4_ros2.control.MulticopterGotoSetpointType(self)
        self.xyz = (x, y, z)
        self.speed = speed
        self.heading = heading

    def update_setpoint(self, dt_s):
        del dt_s
        heading = self.heading if math.isfinite(self.heading) else None
        self.goto.update(self.xyz, heading=heading, max_horizontal_speed=self.speed)


def read_params():
    rclpy.init()
    n = Node("voxl_px4_ros2_goto_waypoint_params")
    for name, default in {"x": 0.5, "y": 0.0, "z": -0.5, "speed": 1.0, "heading": float("nan")}.items():
        n.declare_parameter(name, default)
    vals = [float(n.get_parameter(k).value) for k in ("x", "y", "z", "speed", "heading")]
    n.destroy_node()
    rclpy.shutdown()
    return vals


def main():
    x, y, z, speed, heading = read_params()
    node = px4_ros2.Node("voxl_px4_ros2_goto_waypoint", debug_output=True)
    mode = GotoMode(node, x, y, z, speed, heading)
    assert mode.register()
    node.spin()


if __name__ == "__main__":
    main()

