#!/usr/bin/env python3
import math
import time

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class OffboardWaypoint(Node):
    def __init__(self):
        super().__init__("starling_offboard_waypoint")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.5)
        self.declare_parameter("z", -0.45)
        self.declare_parameter("epsilon_x", 0.08)
        self.declare_parameter("epsilon_y", 0.08)
        self.declare_parameter("epsilon_z", 0.08)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("takeoff_settle_s", 2.0)
        self.declare_parameter("track_timeout_s", 5.0)
        self.declare_parameter("pose_timeout_s", 0.5)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_callback, ODOM_QOS
        )

        self.position = None
        self.last_odom_s = 0.0
        self.state = "WAIT_POSE"
        self.state_start_s = time.time()
        self.target = None
        self.timer = self.create_timer(0.05, self.tick)

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        self.position = [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])]
        self.last_odom_s = time.time()

    def tick(self):
        if not self.odom_fresh():
            self.get_logger().warn("waiting for fresh /fmu/out/vehicle_odometry", throttle_duration_sec=2.0)
            return
        if self.state == "LAND":
            return

        if self.state == "WAIT_POSE":
            self.target = [self.position[0], self.position[1], float(self.get_parameter("z").value)]
            self.transition("PRIME")

        self.publish_setpoint(self.target)

        if self.state == "PRIME" and self.elapsed() >= float(self.get_parameter("prime_s").value):
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.transition("TAKEOFF")

        elif self.state == "TAKEOFF" and self.elapsed() >= float(self.get_parameter("takeoff_settle_s").value):
            self.target = [
                float(self.get_parameter("x").value),
                float(self.get_parameter("y").value),
                float(self.get_parameter("z").value),
            ]
            self.get_logger().info(
                f"tracking NED setpoint x={self.target[0]:.2f}, y={self.target[1]:.2f}, z={self.target[2]:.2f}"
            )
            self.transition("TRACK")

        elif self.state == "TRACK" and self.at_target(self.target):
            self.get_logger().info("target reached; landing")
            self.land()

        elif self.state == "TRACK" and self.elapsed() >= float(self.get_parameter("track_timeout_s").value):
            self.get_logger().warn("target was not reached before timeout; landing")
            self.land()

    def odom_fresh(self):
        return self.position is not None and time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def at_target(self, target):
        return (
            abs(self.position[0] - target[0]) <= float(self.get_parameter("epsilon_x").value)
            and abs(self.position[1] - target[1]) <= float(self.get_parameter("epsilon_y").value)
            and abs(self.position[2] - target[2]) <= float(self.get_parameter("epsilon_z").value)
        )

    def publish_setpoint(self, position):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = position
        setpoint.velocity = [math.nan, math.nan, math.nan]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.nan
        self.setpoint_pub.publish(setpoint)

    def command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def land(self):
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.transition("LAND")

    def transition(self, state):
        self.state = state
        self.state_start_s = time.time()
        self.get_logger().info(f"state -> {state}")

    def elapsed(self):
        return time.time() - self.state_start_s


def main():
    rclpy.init()
    node = OffboardWaypoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
