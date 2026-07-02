#!/usr/bin/env python3
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


CONTROL_LAW_TOPIC = "/llm_vision/dconformal_control_law"
POSE_TOPIC = "/fmu/out/vehicle_odometry"
OWNER_TOPIC = "/llm_vision/offboard_owner"
ODOM_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
CONTROL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def interpolate_sample(samples, t):
    if t <= float(samples[0]["t"]):
        return np.array(samples[0]["x"], dtype=float), np.array(samples[0]["u"], dtype=float)
    if t >= float(samples[-1]["t"]):
        return np.array(samples[-1]["x"], dtype=float), np.array(samples[-1]["u"], dtype=float)
    for index in range(1, len(samples)):
        if float(samples[index]["t"]) >= t:
            before = samples[index - 1]
            after = samples[index]
            span = max(float(after["t"]) - float(before["t"]), 1e-9)
            ratio = (t - float(before["t"])) / span
            x = np.array(before["x"], dtype=float) + (np.array(after["x"], dtype=float) - np.array(before["x"], dtype=float)) * ratio
            u = np.array(before["u"], dtype=float) + (np.array(after["u"], dtype=float) - np.array(before["u"], dtype=float)) * ratio
            return x, u
    return np.array(samples[-1]["x"], dtype=float), np.array(samples[-1]["u"], dtype=float)


class ControlLawExecuter(Node):
    def __init__(self):
        super().__init__("control_law_executer")
        self.declare_parameter("control_law_topic", CONTROL_LAW_TOPIC)
        self.declare_parameter("pose_topic", POSE_TOPIC)
        self.declare_parameter("offboard_owner_topic", OWNER_TOPIC)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("pose_timeout_s", 1.0)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("start_accept_m", 0.75)
        self.declare_parameter("airborne_z_max", -0.05)
        self.declare_parameter("debug", True)

        self.control_law_topic = str(self.get_parameter("control_law_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.offboard_owner_topic = str(self.get_parameter("offboard_owner_topic").value)
        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.owner_pub = self.create_publisher(String, self.offboard_owner_topic, CONTROL_QOS)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.control_sub = self.create_subscription(String, self.control_law_topic, self.control_callback, CONTROL_QOS)

        self.position = None
        self.velocity = None
        self.last_odom_s = 0.0
        self.payload = None
        self.samples = []
        self.k = None
        self.fixed_z = None
        self.command_pos = None
        self.command_vel = None
        self.control_start_s = None
        self.last_tick_s = None
        self.state = "WAIT_CONTROL"

        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / publish_hz, self.tick)
        self.get_logger().info(f"waiting for control law on {self.control_law_topic}; reading odometry from {self.pose_topic}")

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.log_debug(f"ignoring odometry pose_frame={msg.pose_frame}", throttle_duration_sec=5.0)
            return
        self.position = [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])]
        self.velocity = [float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])]
        self.last_odom_s = time.time()

    def control_callback(self, msg):
        if self.state != "WAIT_CONTROL":
            self.log_debug("ignoring new control law because executor is already active")
            return
        try:
            payload = json.loads(msg.data)
            samples = payload["trajectory"]["samples"]
            k = np.array(payload["K"], dtype=float)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"failed to parse control law message: {exc}")
            return
        if not samples:
            self.get_logger().error("control law message has no trajectory samples")
            return
        self.payload = payload
        self.samples = samples
        self.k = k
        goal = payload.get("goal", {})
        self.fixed_z = float(goal.get("z", -0.25))
        self.get_logger().info(f"received control law with {len(samples)} samples")
        self.try_start()

    def try_start(self):
        if self.payload is None or self.position is None:
            return
        if self.position[2] > float(self.get_parameter("airborne_z_max").value):
            self.get_logger().warn(
                f"waiting for takeoff before control law execution: current_z={self.position[2]:.2f}",
                throttle_duration_sec=2.0,
            )
            return
        first = np.array(self.samples[0]["x"], dtype=float)
        start_error = math.hypot(self.position[0] - first[0], self.position[1] - first[1])
        if start_error > float(self.get_parameter("start_accept_m").value):
            self.get_logger().error(f"rejecting control law: start error {start_error:.2f} m exceeds limit")
            self.payload = None
            return
        self.command_pos = [self.position[0], self.position[1], self.fixed_z]
        self.command_vel = [self.velocity[0], self.velocity[1], 0.0]
        self.control_start_s = time.time()
        self.last_tick_s = self.control_start_s
        self.state = "PRIME"
        self.get_logger().info("latched control law; priming offboard setpoints")

    def tick(self):
        if self.position is None:
            self.log_debug(f"waiting for odometry on {self.pose_topic}", throttle_duration_sec=2.0)
            return
        if self.state == "WAIT_CONTROL":
            self.try_start()
            return
        if self.state == "HOLD":
            self.publish_owner()
            self.publish_setpoint(self.command_pos, [0.0, 0.0, 0.0])
            return
        if not self.odom_fresh():
            self.log_debug("odometry is stale; holding last command", throttle_duration_sec=2.0)

        now = time.time()
        elapsed = now - self.control_start_s
        dt = max(1e-3, now - self.last_tick_s)
        self.last_tick_s = now
        state = np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]], dtype=float)
        xhat, uhat = interpolate_sample(self.samples, elapsed)
        control = -self.k @ (state - xhat) + uhat

        self.command_vel[0] += float(control[0]) * dt
        self.command_vel[1] += float(control[1]) * dt
        self.command_pos[0] += self.command_vel[0] * dt
        self.command_pos[1] += self.command_vel[1] * dt

        if self.state == "PRIME" and elapsed >= float(self.get_parameter("prime_s").value):
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.state = "TRACK"
            self.get_logger().info("control law executor switched to TRACK")

        self.publish_owner()
        self.publish_setpoint(self.command_pos, self.command_vel)

        if elapsed >= float(self.samples[-1]["t"]):
            self.state = "HOLD"
            self.command_vel = [0.0, 0.0, 0.0]

    def odom_fresh(self):
        return time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def publish_setpoint(self, position, velocity):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = True
        mode.acceleration = False
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = [float(position[0]), float(position[1]), float(position[2])]
        setpoint.velocity = [float(velocity[0]), float(velocity[1]), 0.0]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.nan
        setpoint.yawspeed = math.nan
        self.setpoint_pub.publish(setpoint)

    def publish_owner(self):
        msg = String()
        msg.data = json.dumps({"owner": "control_law_executer", "timestamp": time.time()})
        self.owner_pub.publish(msg)

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

    def log_debug(self, message, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message, **kwargs)


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print("usage: control_law_executer.py --ros-args --params-file <params.yaml>")
        print("Subscribes to /llm_vision/dconformal_control_law and publishes PX4 position/velocity setpoints.")
        return
    rclpy.init()
    node = ControlLawExecuter()
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
