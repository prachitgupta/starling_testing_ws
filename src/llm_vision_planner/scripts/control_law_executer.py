#!/usr/bin/env python3
"""Own takeoff, verified-plan QP tracking, and landing in PX4 offboard mode."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLandDetected,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

try:
    from min_control_qp import generate_trajectory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from min_control_qp import generate_trajectory


PLAN_TOPIC = "/llm_vision/plan_verified"
MISSION_STATE_TOPIC = "/llm_vision/mission_state"
OWNER_TOPIC = "/llm_vision/offboard_owner"
POSE_TOPIC = "/fmu/out/vehicle_odometry"
STATUS_TOPIC = "/fmu/out/vehicle_status"
LAND_DETECTED_TOPIC = "/fmu/out/vehicle_land_detected"
FEEDBACK_K = np.array(
    [
        [3.1622776601683786, 0.0, 1.7838095742634217, 0.0],
        [0.0, 3.1622776601683786, 0.0, 1.7838095742634217],
    ],
    dtype=float,
)
ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def interpolate_sample(samples, timestamp):
    if timestamp <= float(samples[0]["t"]):
        return np.asarray(samples[0]["x"], dtype=float), np.asarray(samples[0]["u"], dtype=float)
    if timestamp >= float(samples[-1]["t"]):
        return np.asarray(samples[-1]["x"], dtype=float), np.asarray(samples[-1]["u"], dtype=float)
    for index in range(1, len(samples)):
        if float(samples[index]["t"]) >= timestamp:
            before, after = samples[index - 1], samples[index]
            span = max(float(after["t"]) - float(before["t"]), 1e-9)
            ratio = (timestamp - float(before["t"])) / span
            state = np.asarray(before["x"], dtype=float) + ratio * (
                np.asarray(after["x"], dtype=float) - np.asarray(before["x"], dtype=float)
            )
            control = np.asarray(before["u"], dtype=float) + ratio * (
                np.asarray(after["u"], dtype=float) - np.asarray(before["u"], dtype=float)
            )
            return state, control
    return np.asarray(samples[-1]["x"], dtype=float), np.asarray(samples[-1]["u"], dtype=float)


def feedback_control(actual_state, reference_state, reference_control, gain=FEEDBACK_K):
    """Single-score report control law: u=uhat-K(x-xhat)."""
    return np.asarray(reference_control, dtype=float) - np.asarray(gain, dtype=float) @ (
        np.asarray(actual_state, dtype=float) - np.asarray(reference_state, dtype=float)
    )


class ControlLawExecuter(Node):
    def __init__(self):
        super().__init__("control_law_executer")
        self.declare_parameter("plan_topic", PLAN_TOPIC)
        self.declare_parameter("mission_state_topic", MISSION_STATE_TOPIC)
        self.declare_parameter("offboard_owner_topic", OWNER_TOPIC)
        self.declare_parameter("pose_topic", POSE_TOPIC)
        self.declare_parameter("vehicle_status_topic", STATUS_TOPIC)
        self.declare_parameter("land_detected_topic", LAND_DETECTED_TOPIC)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("pose_timeout_s", 1.0)
        self.declare_parameter("status_timeout_s", 2.0)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("takeoff_z", -0.25)
        self.declare_parameter("takeoff_accept_m", 0.08)
        self.declare_parameter("takeoff_settle_s", 2.0)
        self.declare_parameter("hover_speed_accept_mps", 0.10)
        self.declare_parameter("start_accept_m", 0.10)
        self.declare_parameter("goal_accept_m", 0.20)
        self.declare_parameter("goal_settle_s", 2.0)
        self.declare_parameter("command_retry_s", 1.0)
        self.declare_parameter("transition_timeout_s", 15.0)
        self.declare_parameter("trajectory_dt", 0.1)
        self.declare_parameter("debug", True)

        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.owner_topic = str(self.get_parameter("offboard_owner_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.status_topic = str(self.get_parameter("vehicle_status_topic").value)
        self.land_detected_topic = str(self.get_parameter("land_detected_topic").value)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.mission_state_pub = self.create_publisher(String, self.mission_state_topic, 10)
        self.owner_pub = self.create_publisher(String, self.owner_topic, LATCHED_QOS)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.status_sub = self.create_subscription(VehicleStatus, self.status_topic, self.status_callback, ODOM_QOS)
        self.land_sub = self.create_subscription(
            VehicleLandDetected,
            self.land_detected_topic,
            self.land_callback,
            ODOM_QOS,
        )
        self.plan_sub = self.create_subscription(String, self.plan_topic, self.plan_callback, LATCHED_QOS)

        self.position = None
        self.velocity = None
        self.yaw = math.nan
        self.last_odom_s = 0.0
        self.vehicle_status = None
        self.last_status_s = 0.0
        self.landed = None
        self.ground_z = None
        self.takeoff_target = None
        self.goal_target = None
        self.samples = []
        self.track_start_s = None
        self.dwell_start_s = None
        self.last_command_s = -math.inf
        self.last_setpoint_position = None
        self.last_setpoint_velocity = None
        self.state = "WAIT_ODOMETRY"
        self.state_start_s = time.monotonic()
        self.failure_reason = None

        publish_hz = max(3.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / publish_hz, self.tick)
        self.get_logger().info(
            f"unified offboard executor waiting for odometry on {self.pose_topic} and verified plans on {self.plan_topic}"
        )

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.log_debug(f"ignoring non-NED odometry pose_frame={msg.pose_frame}", throttle_duration_sec=5.0)
            return
        self.position = [float(value) for value in msg.position]
        self.velocity = [float(value) for value in msg.velocity]
        self.yaw = self.quat_to_yaw(float(msg.q[1]), float(msg.q[2]), float(msg.q[3]), float(msg.q[0]))
        self.last_odom_s = time.monotonic()

    def status_callback(self, msg):
        self.vehicle_status = msg
        self.last_status_s = time.monotonic()

    def land_callback(self, msg):
        self.landed = bool(msg.landed)

    def plan_callback(self, msg):
        if self.state != "HOLDING_FOR_PLAN":
            self.log_debug(f"ignoring verified plan while state={self.state}")
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"failed to parse verified plan: {exc}")
            self.fail_or_land("invalid verified-plan JSON")
            return
        if not payload.get("passed", False):
            self.get_logger().warning(
                f"failed verified plan; constraints={payload.get('failed_constraints', [])}"
            )
            self.fail_or_land("verified plan failed")
            return
        try:
            waypoints = self.valid_waypoints(payload.get("waypoints", []))
        except ValueError as exc:
            self.get_logger().error(f"invalid verified plan: {exc}")
            self.fail_or_land("invalid verified plan")
            return
        start_error = math.sqrt(
            (waypoints[0]["x"] - self.position[0]) ** 2
            + (waypoints[0]["y"] - self.position[1]) ** 2
            + (waypoints[0]["z"] - self.position[2]) ** 2
        )
        if start_error > float(self.get_parameter("start_accept_m").value):
            self.get_logger().error(
                f"rejecting verified plan: start error {start_error:.3f} m exceeds configured limit"
            )
            self.fail_or_land("verified plan start mismatch")
            return
        try:
            trajectory = generate_trajectory(
                waypoints,
                payload.get("workspace", {}),
                payload.get("obstacles", []),
                dt=float(self.get_parameter("trajectory_dt").value),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().error(f"minimum-control QP failed: {exc}")
            self.fail_or_land("minimum-control QP failure")
            return
        self.samples = trajectory["samples"]
        self.goal_target = [waypoints[-1]["x"], waypoints[-1]["y"], waypoints[-1]["z"]]
        self.track_start_s = time.monotonic()
        self.transition("TRACK_QP")
        self.get_logger().info(
            f"accepted plan_id={payload.get('plan_id')} with {len(waypoints)} waypoints and {len(self.samples)} QP samples"
        )

    def tick(self):
        now = time.monotonic()
        if self.state == "WAIT_ODOMETRY":
            if self.odom_fresh(now) and self.status_fresh(now):
                self.ground_z = self.position[2]
                self.takeoff_target = [
                    self.position[0],
                    self.position[1],
                    float(self.get_parameter("takeoff_z").value),
                ]
                self.set_hold(self.takeoff_target)
                self.transition("PRIME_OFFBOARD")
            self.publish_mission_state()
            return

        if self.state not in ("COMPLETE", "FAILED", "LAND"):
            if not self.odom_fresh(now):
                self.fail_or_land("odometry timeout")
            elif not self.status_fresh(now):
                self.fail_or_land("vehicle-status timeout")
            elif self.vehicle_status is not None and bool(self.vehicle_status.failsafe):
                self.fail_or_land("PX4 entered failsafe")

        if self.state == "PRIME_OFFBOARD":
            self.set_hold(self.takeoff_target)
            if self.elapsed(now) >= float(self.get_parameter("prime_s").value):
                self.request_offboard_and_arm(now)
                self.transition("ARM_TAKEOFF")

        elif self.state == "ARM_TAKEOFF":
            self.set_hold(self.takeoff_target)
            if self.armed_and_offboard():
                self.transition("TAKEOFF")
            elif self.elapsed(now) >= float(self.get_parameter("transition_timeout_s").value):
                self.fail_or_land("arming/offboard confirmation timeout")
            else:
                self.request_offboard_and_arm(now)

        elif self.state == "TAKEOFF":
            self.set_hold(self.takeoff_target)
            if self.elapsed(now) >= float(self.get_parameter("transition_timeout_s").value):
                self.fail_or_land("takeoff confirmation timeout")
            elif self.within_target(self.takeoff_target, float(self.get_parameter("takeoff_accept_m").value)):
                if self.speed() <= float(self.get_parameter("hover_speed_accept_mps").value):
                    self.update_dwell(now)
                    if now - self.dwell_start_s >= float(self.get_parameter("takeoff_settle_s").value):
                        self.transition("HOLDING_FOR_PLAN")
                else:
                    self.dwell_start_s = None
            else:
                self.dwell_start_s = None

        elif self.state == "HOLDING_FOR_PLAN":
            self.set_hold(self.takeoff_target)

        elif self.state == "TRACK_QP":
            elapsed = now - self.track_start_s
            reference_state, reference_control = interpolate_sample(self.samples, elapsed)
            actual_state = [self.position[0], self.position[1], self.velocity[0], self.velocity[1]]
            command = feedback_control(actual_state, reference_state, reference_control)
            self.last_setpoint_position = [math.nan, math.nan, self.takeoff_target[2]]
            self.last_setpoint_velocity = [float(command[0]), float(command[1]), 0.0]
            if elapsed >= float(self.samples[-1]["t"]):
                self.set_hold(self.goal_target)
                self.transition("GOAL_HOLD")

        elif self.state == "GOAL_HOLD":
            self.set_hold(self.goal_target)
            if self.elapsed(now) >= float(self.get_parameter("transition_timeout_s").value):
                self.fail_or_land("goal confirmation timeout")
            elif self.within_target(self.goal_target, float(self.get_parameter("goal_accept_m").value)):
                if self.speed() <= float(self.get_parameter("hover_speed_accept_mps").value):
                    self.update_dwell(now)
                    if now - self.dwell_start_s >= float(self.get_parameter("goal_settle_s").value):
                        self.begin_land("goal confirmed")
                else:
                    self.dwell_start_s = None
            else:
                self.dwell_start_s = None

        elif self.state == "LAND":
            if self.landed is True:
                if self.is_armed():
                    self.request_disarm(now)
                else:
                    self.transition("COMPLETE")
            else:
                self.request_land(now)

        if self.state not in ("COMPLETE", "FAILED") and self.last_setpoint_position is not None:
            self.publish_owner()
            self.publish_setpoint(self.last_setpoint_position, self.last_setpoint_velocity)
        self.publish_mission_state()

    def valid_waypoints(self, raw_waypoints):
        if len(raw_waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        waypoints = []
        for waypoint in raw_waypoints:
            if not all(key in waypoint for key in ("x", "y", "z")):
                raise ValueError("every waypoint requires x, y, and z")
            parsed = {key: float(waypoint[key]) for key in ("x", "y", "z")}
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError("waypoint values must be finite")
            waypoints.append(parsed)
        return waypoints

    def request_offboard_and_arm(self, now):
        if now - self.last_command_s < float(self.get_parameter("command_retry_s").value):
            return
        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.last_command_s = now

    def begin_land(self, reason):
        if self.state == "LAND":
            return
        if self.position is not None:
            self.set_hold([self.position[0], self.position[1], self.position[2]])
        self.failure_reason = reason if reason != "goal confirmed" else None
        self.transition("LAND")
        self.last_command_s = -math.inf
        self.request_land(time.monotonic())
        self.get_logger().warning(f"PX4 auto-land requested: {reason}")

    def request_land(self, now):
        if now - self.last_command_s < float(self.get_parameter("command_retry_s").value):
            return
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.last_command_s = now

    def request_disarm(self, now):
        if now - self.last_command_s < float(self.get_parameter("command_retry_s").value):
            return
        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.last_command_s = now

    def fail_or_land(self, reason):
        if self.airborne():
            self.begin_land(reason)
        else:
            self.failure_reason = reason
            self.transition("FAILED")
            self.get_logger().error(reason)

    def armed_and_offboard(self):
        return bool(
            self.vehicle_status is not None
            and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

    def is_armed(self):
        return bool(
            self.vehicle_status is not None
            and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )

    def airborne(self):
        if self.is_armed() and not self.landed:
            return True
        return bool(
            self.position is not None
            and self.ground_z is not None
            and self.position[2] < self.ground_z - 0.05
        )

    def odom_fresh(self, now=None):
        now = time.monotonic() if now is None else now
        return self.position is not None and now - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def status_fresh(self, now=None):
        now = time.monotonic() if now is None else now
        return self.vehicle_status is not None and now - self.last_status_s <= float(
            self.get_parameter("status_timeout_s").value
        )

    def set_hold(self, position):
        self.last_setpoint_position = [float(value) for value in position]
        self.last_setpoint_velocity = [0.0, 0.0, 0.0]

    def within_target(self, target, tolerance):
        return math.sqrt(sum((self.position[index] - target[index]) ** 2 for index in range(3))) <= tolerance

    def speed(self):
        return math.sqrt(sum(value * value for value in self.velocity))

    def update_dwell(self, now):
        if self.dwell_start_s is None:
            self.dwell_start_s = now

    def transition(self, state):
        self.state = state
        self.state_start_s = time.monotonic()
        self.dwell_start_s = None
        self.get_logger().info(f"state -> {state}")

    def elapsed(self, now=None):
        now = time.monotonic() if now is None else now
        return now - self.state_start_s

    def publish_setpoint(self, position, velocity):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = True
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        mode.thrust_and_torque = False
        mode.direct_actuator = False
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = [float(value) for value in position]
        setpoint.velocity = [float(value) for value in velocity]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.nan
        setpoint.yawspeed = math.nan
        self.setpoint_pub.publish(setpoint)

    def publish_mission_state(self):
        msg = String()
        msg.data = json.dumps(
            {
                "state": self.state,
                "position": {
                    "x": self.position[0] if self.position else None,
                    "y": self.position[1] if self.position else None,
                    "z": self.position[2] if self.position else None,
                },
                "heading_deg": math.degrees(self.yaw) if math.isfinite(self.yaw) else None,
                "failure_reason": self.failure_reason,
                "timestamp": time.time(),
            }
        )
        self.mission_state_pub.publish(msg)

    def publish_owner(self):
        msg = String()
        msg.data = json.dumps({"owner": "control_law_executer", "state": self.state, "timestamp": time.time()})
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

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def log_debug(self, message, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message, **kwargs)


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print("usage: control_law_executer.py --ros-args --params-file <params.yaml>")
        print("Owns PX4 takeoff, verified-waypoint QP tracking, and auto-land.")
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
