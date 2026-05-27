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


def bezier(p0, p3, t):
    p1 = p0
    p2 = p3
    u = 1.0 - t
    return [
        u**3 * p0[i] + 3.0 * u**2 * t * p1[i] + 3.0 * u * t**2 * p2[i] + t**3 * p3[i]
        for i in range(3)
    ]


def bezier_velocity(p0, p3, t, duration_s):
    p1 = p0
    p2 = p3
    u = 1.0 - t
    return [
        (
            3.0 * u**2 * (p1[i] - p0[i])
            + 6.0 * u * t * (p2[i] - p1[i])
            + 3.0 * t**2 * (p3[i] - p2[i])
        )
        / duration_s
        for i in range(3)
    ]


class BezierOffboardWaypoint(Node):
    def __init__(self):
        super().__init__("starling_bezier_offboard_waypoint")
        self.declare_parameter("x", 0.7)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("x2", 0.7)
        self.declare_parameter("y2", -0.3)
        self.declare_parameter("z", -0.25)
        self.declare_parameter("epsilon_x", 0.1)
        self.declare_parameter("epsilon_y", 0.1)
        self.declare_parameter("epsilon_z", 0.1)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("takeoff_settle_s", 2.0)
        self.declare_parameter("takeoff_accept_m", 0.08)
        self.declare_parameter("duration_s", 8.0)
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
        self.takeoff_target = None
        self.hold_setpoint = None
        self.start = None
        self.goal = None
        self.goals = []
        self.goal_index = 0
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

        position_setpoint = self.hold_setpoint or self.takeoff_target or self.position
        velocity_setpoint = [math.nan, math.nan, math.nan]

        if self.state == "WAIT_POSE":
            self.takeoff_target = [self.position[0], self.position[1], float(self.get_parameter("z").value)]
            self.transition("PRIME")

        elif self.state == "PRIME" and self.elapsed() >= float(self.get_parameter("prime_s").value):
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.transition("TAKEOFF")

        elif self.state == "TAKEOFF" and self.takeoff_settled():
            self.start = list(self.position)
            self.goal = [
                float(self.get_parameter("x").value),
                float(self.get_parameter("y").value),
                float(self.get_parameter("z").value),
            ]
            self.goals = [
                self.goal,
                [
                    float(self.get_parameter("x2").value),
                    float(self.get_parameter("y2").value),
                    float(self.get_parameter("z").value),
                ],
            ]
            self.goal_index = 0
            self.transition("TRACK")

        elif self.state == "TRACK":
            duration_s = max(0.5, float(self.get_parameter("duration_s").value))
            t = min(1.0, self.elapsed() / duration_s)
            position_setpoint = bezier(self.start, self.goal, t)
            velocity_setpoint = bezier_velocity(self.start, self.goal, t, duration_s)
            if t >= 1.0 and self.at_target(self.goal):
                if self.goal_index + 1 < len(self.goals):
                    self.goal_index += 1
                    self.start = list(self.position)
                    self.goal = self.goals[self.goal_index]
                    self.transition("TRACK")
                else:
                    position_setpoint, velocity_setpoint = self.finish_mission()
            if self.elapsed() >= duration_s + float(self.get_parameter("track_timeout_s").value):
                self.get_logger().warn("Bezier target was not reached before timeout; landing")
                self.land()
                return

        elif self.state == "HOLD":
            position_setpoint = self.hold_setpoint
            velocity_setpoint = [0.0, 0.0, 0.0]

        self.publish_setpoint(position_setpoint, velocity_setpoint)

    def odom_fresh(self):
        return self.position is not None and time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def at_target(self, target):
        return (
            abs(self.position[0] - target[0]) <= float(self.get_parameter("epsilon_x").value)
            and abs(self.position[1] - target[1]) <= float(self.get_parameter("epsilon_y").value)
            and abs(self.position[2] - target[2]) <= float(self.get_parameter("epsilon_z").value)
        )

    def takeoff_settled(self):
        if self.takeoff_target is None:
            return False
        z_error = abs(self.position[2] - self.takeoff_target[2])
        return (
            z_error <= float(self.get_parameter("takeoff_accept_m").value)
            and self.elapsed() >= float(self.get_parameter("takeoff_settle_s").value)
        )

    def finish_mission(self):
        self.hold_setpoint = list(self.position) if self.position is not None else list(self.goal)
        self.get_logger().info("final waypoint reached; holding position for RC landing")
        self.transition("HOLD")
        return self.hold_setpoint, [0.0, 0.0, 0.0]

    def publish_setpoint(self, position, velocity):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = position
        setpoint.velocity = velocity
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
    node = BezierOffboardWaypoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
