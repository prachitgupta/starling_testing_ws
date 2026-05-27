#!/usr/bin/env python3
import json
import math
import time

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
VERIFIED_PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


class ContinuousTrajectoryFollower(Node):
    def __init__(self):
        super().__init__("llm_vision_trajectory_follower_continuous")
        self.declare_parameter("plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("offboard_owner_topic", "/llm_vision/offboard_owner")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("accept_m", 0.35)
        self.declare_parameter("start_accept_m", 0.75)
        self.declare_parameter("pose_timeout_s", 1.0)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("debug", True)

        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.offboard_owner_topic = str(self.get_parameter("offboard_owner_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.owner_pub = self.create_publisher(String, self.offboard_owner_topic, VERIFIED_PLAN_QOS)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.plan_sub = self.create_subscription(String, self.plan_topic, self.plan_callback, VERIFIED_PLAN_QOS)

        self.position = None
        self.last_odom_s = 0.0
        self.state = "WAIT_PLAN"
        self.state_start_s = time.time()
        self.waypoints = []
        self.pending_waypoints = None
        self.waypoint_index = 0
        self.hold_setpoint = None

        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / publish_hz, self.tick)
        self.get_logger().info(
            f"waiting for verified plan on {self.plan_topic}; reading odometry from {self.pose_topic}"
        )

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.log_debug(
                f"ignoring odometry with pose_frame={msg.pose_frame}; expected {VehicleOdometry.POSE_FRAME_NED}",
                throttle_duration_sec=5.0,
            )
            return
        self.position = [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])]
        self.last_odom_s = time.time()

    def plan_callback(self, msg):
        self.get_logger().info(f"received verified plan message ({len(msg.data)} bytes)")
        if self.state != "WAIT_PLAN":
            self.log_debug("ignoring verified plan because tracker is already active")
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"failed to parse verified plan JSON: {exc}")
            return

        if not payload.get("passed", False):
            self.get_logger().warn(
                f"ignoring verified plan because passed=false; failed_constraints={payload.get('failed_constraints', [])}"
            )
            return

        waypoints = []
        for waypoint in payload.get("waypoints", []):
            if all(k in waypoint for k in ("x", "y", "z")):
                waypoints.append([float(waypoint["x"]), float(waypoint["y"]), float(waypoint["z"])])

        if not waypoints:
            self.get_logger().warn("ignoring verified plan with no usable waypoints")
            return
        if self.position is None:
            self.pending_waypoints = waypoints
            self.get_logger().warn("verified plan arrived before odometry; will latch it after first odometry update")
            return

        self.latch_waypoints(waypoints)

    def latch_waypoints(self, waypoints):
        if self.position is None:
            return False

        start_error = norm(sub(waypoints[0], self.position))
        start_accept_m = float(self.get_parameter("start_accept_m").value)
        if start_error > start_accept_m:
            self.get_logger().error(
                f"rejecting verified plan: first waypoint is {start_error:.2f} m away; "
                f"limit is {start_accept_m:.2f} m; "
                f"current_position={self.position}; first_waypoint={waypoints[0]}"
            )
            return False

        self.waypoints = waypoints
        self.pending_waypoints = None
        self.waypoint_index = 0
        self.hold_setpoint = list(self.position)
        self.transition("PRIME")
        self.get_logger().info(f"latched waypoint trajectory with {len(self.waypoints)} waypoints")
        return True

    def tick(self):
        if self.state == "WAIT_PLAN":
            if self.pending_waypoints is not None:
                self.latch_waypoints(self.pending_waypoints)
            return

        if self.position is None:
            self.log_debug(f"waiting for PX4 odometry on {self.pose_topic}", throttle_duration_sec=2.0)
            return

        if not self.odom_fresh():
            self.log_debug("PX4 odometry is stale; continuing with last received pose", throttle_duration_sec=2.0)

        target = self.hold_setpoint or self.position

        if self.state == "PRIME":
            if self.elapsed() >= float(self.get_parameter("prime_s").value):
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.publish_owner()
                self.transition("TRACK")

        elif self.state == "TRACK":
            target = self.current_target()
            if self.at_target(target):
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    self.hold_setpoint = list(self.waypoints[-1])
                    self.get_logger().info("goal reached yay; holding final waypoint for RC landing")
                    self.transition("HOLD")
                else:
                    target = self.current_target()
                    self.log_debug(f"advancing to waypoint {self.waypoint_index + 1}/{len(self.waypoints)}")

        elif self.state == "HOLD":
            target = self.hold_setpoint

        if self.state in ("TRACK", "HOLD"):
            self.publish_owner()
        self.publish_setpoint(target)

    def current_target(self):
        return self.waypoints[self.waypoint_index]

    def at_target(self, target):
        accept_m = float(self.get_parameter("accept_m").value)
        return (
            abs(self.position[0] - target[0]) <= accept_m
            and abs(self.position[1] - target[1]) <= accept_m
            and abs(self.position[2] - target[2]) <= accept_m
        )

    def odom_fresh(self):
        return self.position is not None and time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def publish_setpoint(self, position):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = position
        setpoint.velocity = [math.nan, math.nan, math.nan]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.nan
        setpoint.yawspeed = math.nan
        self.setpoint_pub.publish(setpoint)

    def publish_owner(self):
        msg = String()
        msg.data = json.dumps({"owner": "trajectory_follower_continuous", "timestamp": time.time()})
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

    def transition(self, state):
        self.state = state
        self.state_start_s = time.time()
        self.log_debug(f"state -> {state}")

    def elapsed(self):
        return time.time() - self.state_start_s

    def log_debug(self, message, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message, **kwargs)


def main():
    rclpy.init()
    node = ContinuousTrajectoryFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
