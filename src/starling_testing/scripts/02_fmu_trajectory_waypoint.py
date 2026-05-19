#!/usr/bin/env python3
import math
import time

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

ODOMETRY_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class TrajectoryWaypoint(Node):
    def __init__(self):
        super().__init__("voxl_fmu_trajectory_waypoint")
        self.declare_parameter("x", 0.5)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", -0.5)
        self.declare_parameter("arm_and_offboard", False)
        self.declare_parameter("move_after_s", 3.0)
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("pose_timeout_s", 0.5)
        self.offboard = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.pose_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.pose_callback, ODOMETRY_QOS)
        self.count = 0
        self.odometry = None
        self.hold_xy = None
        self.hold_heading = float("nan")
        self.create_timer(0.05, self.tick)

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.get_logger().warn(
                f"ignoring {self.pose_topic}: pose_frame={msg.pose_frame}, expected NED",
                throttle_duration_sec=2.0,
            )
            return
        self.odometry = {
            "x": float(msg.position[0]),
            "y": float(msg.position[1]),
            "z": float(msg.position[2]),
            "yaw": self.quat_to_yaw(msg.q[1], msg.q[2], msg.q[3], msg.q[0]),
            "stamp": time.time(),
        }
        if self.hold_xy is None:
            self.hold_xy = (self.odometry["x"], self.odometry["y"])
            self.hold_heading = self.odometry["yaw"]
            self.get_logger().info(
                f"holding current odometry XY before move: x={self.hold_xy[0]:.2f}, y={self.hold_xy[1]:.2f}")

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odometry_fresh(self):
        if self.odometry is None:
            return False
        timeout_s = float(self.get_parameter("pose_timeout_s").value)
        return time.time() - self.odometry["stamp"] <= timeout_s

    def vehicle_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = msg.source_system = 1
        msg.target_component = msg.source_component = 1
        msg.from_external = True
        self.command.publish(msg)

    def tick(self):
        if not self.odometry_fresh() or self.hold_xy is None:
            self.get_logger().warn(f"waiting for fresh PX4 odometry on {self.pose_topic}", throttle_duration_sec=2.0)
            return

        now = int(self.get_clock().now().nanoseconds / 1000)
        hb = OffboardControlMode()
        hb.timestamp = now
        hb.position = True
        self.offboard.publish(hb)

        move_after_ticks = max(20, int(float(self.get_parameter("move_after_s").value) / 0.05))
        if self.count < move_after_ticks:
            x, y = self.hold_xy
            z = float(self.get_parameter("z").value)
        else:
            x = float(self.get_parameter("x").value)
            y = float(self.get_parameter("y").value)
            z = float(self.get_parameter("z").value)
            if self.count == move_after_ticks:
                self.get_logger().info(f"moving to waypoint: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        sp = TrajectorySetpoint()
        sp.timestamp = now
        sp.position = [x, y, z]
        sp.yaw = self.hold_heading
        self.setpoint.publish(sp)

        if self.count == 20 and bool(self.get_parameter("arm_and_offboard").value):
            self.get_logger().info("requesting Offboard mode and arm")
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.count += 1


def main():
    rclpy.init()
    node = TrajectoryWaypoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
