#!/usr/bin/env python3
import math
import time

import px4_ros2
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

ODOMETRY_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class GotoMode(px4_ros2.components.ModeBase):
    def __init__(self, node, x, y, z, speed, heading, pose_topic, pose_timeout_s):
        super().__init__(node=node, mode_name="VOXL Goto Smoke Test")
        self.node = node
        self.goto = px4_ros2.control.MulticopterGotoSetpointType(self)
        self.xyz = (x, y, z)
        self.speed = speed
        self.heading = heading
        self.pose_topic = pose_topic
        self.pose_timeout_s = pose_timeout_s
        self.last_odometry_s = None
        self.pose_sub = node.create_subscription(VehicleOdometry, pose_topic, self.pose_callback, ODOMETRY_QOS)

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.node.get_logger().warning(
                f"ignoring {self.pose_topic}: pose_frame={msg.pose_frame}, expected NED",
                throttle_duration_sec=2.0,
            )
            return
        self.last_odometry_s = time.time()

    def odometry_fresh(self):
        return self.last_odometry_s is not None and time.time() - self.last_odometry_s <= self.pose_timeout_s

    def update_setpoint(self, dt_s):
        del dt_s
        if not self.odometry_fresh():
            self.node.get_logger().warning(
                f"waiting for fresh PX4 odometry on {self.pose_topic}", throttle_duration_sec=2.0)
            return
        heading = self.heading if math.isfinite(self.heading) else None
        self.goto.update(self.xyz, heading=heading, max_horizontal_speed=self.speed)


def read_params():
    rclpy.init()
    n = Node("voxl_px4_ros2_goto_waypoint_params")
    defaults = {
        "x": 0.3,
        "y": 0.0,
        "z": -0.2,
        "speed": 1.0,
        "heading": float("nan"),
        "pose_topic": "/fmu/out/vehicle_odometry",
        "pose_timeout_s": 0.5,
    }
    for name, default in defaults.items():
        n.declare_parameter(name, default)
    vals = [float(n.get_parameter(k).value) for k in ("x", "y", "z", "speed", "heading")]
    vals.extend([str(n.get_parameter("pose_topic").value), float(n.get_parameter("pose_timeout_s").value)])
    n.destroy_node()
    rclpy.shutdown()
    return vals


def main():
    x, y, z, speed, heading, pose_topic, pose_timeout_s = read_params()
    node = px4_ros2.Node("voxl_px4_ros2_goto_waypoint", debug_output=True)
    mode = GotoMode(node, x, y, z, speed, heading, pose_topic, pose_timeout_s)
    assert mode.register()
    node.spin()


if __name__ == "__main__":
    main()
