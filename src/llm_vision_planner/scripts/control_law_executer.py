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
)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

try:
    from min_control_qp import evaluate_sample, generate_trajectory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from min_control_qp import evaluate_sample, generate_trajectory


PLAN_TOPIC = "/llm_vision/plan_verified"
MISSION_STATE_TOPIC = "/llm_vision/mission_state"
OWNER_TOPIC = "/llm_vision/offboard_owner"
EXECUTOR_COMMAND_TOPIC = "/llm_vision/executor_command"
POSE_TOPIC = "/fmu/out/vehicle_odometry"
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


def feedback_control(actual_state, reference_state, reference_control, gain=FEEDBACK_K):
    """Calibrated planar control law: u=uhat-K(x-xhat)."""
    return np.asarray(reference_control, dtype=float) - np.asarray(gain, dtype=float) @ (
        np.asarray(actual_state, dtype=float) - np.asarray(reference_state, dtype=float)
    )


class ControlLawExecuter(Node):
    def __init__(self):
        super().__init__("control_law_executer")
        self.declare_parameter("plan_topic", PLAN_TOPIC)
        self.declare_parameter("mission_state_topic", MISSION_STATE_TOPIC)
        self.declare_parameter("offboard_owner_topic", OWNER_TOPIC)
        self.declare_parameter("executor_command_topic", EXECUTOR_COMMAND_TOPIC)
        self.declare_parameter("pose_topic", POSE_TOPIC)
        self.declare_parameter("land_detected_topic", LAND_DETECTED_TOPIC)
        self.declare_parameter("takeoff_gate_topic", "")
        self.declare_parameter("takeoff_gate_status", "FRAME_READY")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("pose_timeout_s", 1.0)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("takeoff_z", -0.5)
        self.declare_parameter("takeoff_accept_m", 0.08)
        self.declare_parameter("takeoff_settle_s", 2.0)
        self.declare_parameter("hover_speed_accept_mps", 0.10)
        self.declare_parameter("land_descent_speed_mps", 0.30)
        self.declare_parameter("start_accept_m", 0.10)
        self.declare_parameter("goal_accept_m", 0.20)
        self.declare_parameter("goal_settle_s", 2.0)
        self.declare_parameter("land_after_complete", True)
        self.declare_parameter("command_retry_s", 1.0)
        self.declare_parameter("transition_timeout_s", 15.0)
        self.declare_parameter("trajectory_dt", 0.1)
        self.declare_parameter("max_horizontal_speed_mps", 0.5)
        self.declare_parameter("max_horizontal_acceleration_mps2", 0.5)
        self.declare_parameter("debug", True)

        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.owner_topic = str(self.get_parameter("offboard_owner_topic").value)
        self.executor_command_topic = str(self.get_parameter("executor_command_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.land_detected_topic = str(self.get_parameter("land_detected_topic").value)
        self.takeoff_gate_topic = str(self.get_parameter("takeoff_gate_topic").value).strip()
        self.takeoff_gate_status = str(self.get_parameter("takeoff_gate_status").value).strip()
        self.takeoff_gate_ready = not self.takeoff_gate_topic
        self.land_after_complete = bool(self.get_parameter("land_after_complete").value)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.mission_state_pub = self.create_publisher(String, self.mission_state_topic, 10)
        self.owner_pub = self.create_publisher(String, self.owner_topic, LATCHED_QOS)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.land_sub = self.create_subscription(
            VehicleLandDetected,
            self.land_detected_topic,
            self.land_callback,
            ODOM_QOS,
        )
        self.plan_sub = self.create_subscription(String, self.plan_topic, self.plan_callback, LATCHED_QOS)
        self.executor_command_sub = self.create_subscription(
            String,
            self.executor_command_topic,
            self.executor_command_callback,
            10,
        )
        self.takeoff_gate_sub = None
        if self.takeoff_gate_topic:
            self.takeoff_gate_sub = self.create_subscription(
                String,
                self.takeoff_gate_topic,
                self.takeoff_gate_callback,
                LATCHED_QOS,
            )

        self.position = None
        self.velocity = None
        self.yaw = math.nan
        self.last_odom_s = 0.0
        self.landed = None
        self.ground_z = None
        self.takeoff_target = None
        self.goal_target = None
        self.landing_target = None
        self.samples = []
        self.track_start_s = None
        self.dwell_start_s = None
        self.last_command_s = -math.inf
        self.last_setpoint_position = None
        self.last_setpoint_velocity = None
        self.last_track_command = None
        self.last_track_command_s = None
        self.state = "WAIT_ODOMETRY"
        self.state_start_s = time.monotonic()
        self.failure_reason = None

        publish_hz = max(3.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / publish_hz, self.tick)
        self.get_logger().info(
            f"unified offboard executor waiting for odometry on {self.pose_topic} and verified plans on {self.plan_topic}"
        )

    def takeoff_gate_callback(self, msg):
        if self.takeoff_gate_ready:
            return
        try:
            status = str(json.loads(msg.data).get("status", "")).strip()
        except (AttributeError, json.JSONDecodeError):
            return
        if status == self.takeoff_gate_status:
            self.takeoff_gate_ready = True
            self.get_logger().info(
                f"takeoff gate opened by status={status} on {self.takeoff_gate_topic}"
            )

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.log_debug(f"ignoring non-NED odometry pose_frame={msg.pose_frame}", throttle_duration_sec=5.0)
            return
        self.position = [float(value) for value in msg.position]
        self.velocity = [float(value) for value in msg.velocity]
        self.yaw = self.quat_to_yaw(float(msg.q[1]), float(msg.q[2]), float(msg.q[3]), float(msg.q[0]))
        self.last_odom_s = time.monotonic()

    def land_callback(self, msg):
        self.landed = bool(msg.landed)

    def executor_command_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"failed to parse executor command: {exc}")
            return
        if str(payload.get("command", "")).strip().upper() != "LAND":
            self.get_logger().warning(f"ignoring unsupported executor command: {payload.get('command')}")
            return
        if self.state in ("LAND", "COMPLETE", "FAILED"):
            return
        self.begin_land(str(payload.get("reason") or "operator requested landing"))

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
                max_velocity_mps=float(self.get_parameter("max_horizontal_speed_mps").value),
                max_acceleration_mps2=float(self.get_parameter("max_horizontal_acceleration_mps2").value),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().error(f"minimum-control QP failed: {exc}")
            self.fail_or_land("minimum-control QP failure")
            return
        self.samples = trajectory["samples"]
        self.goal_target = [waypoints[-1]["x"], waypoints[-1]["y"], waypoints[-1]["z"]]
        self.track_start_s = time.monotonic()
        self.last_track_command = None
        self.last_track_command_s = None
        self.transition("TRACK_QP")
        self.get_logger().info(
            f"accepted plan_id={payload.get('plan_id')} with {len(waypoints)} waypoints and {len(self.samples)} QP samples"
        )

    def tick(self):
        now = time.monotonic()
        if self.state == "WAIT_ODOMETRY":
            if self.odom_fresh(now):
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

        if self.state == "PRIME_OFFBOARD":
            self.set_hold(self.takeoff_target)
            if (
                self.elapsed(now) >= float(self.get_parameter("prime_s").value)
                and self.takeoff_gate_ready
            ):
                self.request_offboard_and_arm(now)
                self.transition("ARM_TAKEOFF")

        elif self.state == "ARM_TAKEOFF":
            self.set_hold(self.takeoff_target)
            if self.elapsed(now) >= float(self.get_parameter("command_retry_s").value):
                self.transition("TAKEOFF")
            elif self.elapsed(now) >= float(self.get_parameter("transition_timeout_s").value):
                self.fail_or_land("arming/offboard command timeout")
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
            reference_state, reference_control = evaluate_sample(self.samples, elapsed)
            actual_state = [self.position[0], self.position[1], self.velocity[0], self.velocity[1]]
            command = self.limit_tracking_command(feedback_control(actual_state, reference_state, reference_control), now)
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
                        if self.land_after_complete:
                            self.begin_land("goal confirmed")
                        else:
                            self.transition("HOLDING_AT_GOAL")
                else:
                    self.dwell_start_s = None
            else:
                self.dwell_start_s = None

        elif self.state == "HOLDING_AT_GOAL":
            self.set_hold(self.goal_target)

        elif self.state == "LAND":
            if self.landed is True:
                self.last_command_s = -math.inf
                self.request_disarm(now)
                self.transition("COMPLETE")

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
        if self.position is None:
            self.failure_reason = reason
            self.transition("FAILED")
            self.get_logger().error(f"cannot start Offboard landing without a local position: {reason}")
            return
        descent_speed = float(self.get_parameter("land_descent_speed_mps").value)
        if not math.isfinite(descent_speed) or descent_speed <= 0.0:
            self.failure_reason = "invalid landing descent speed"
            self.transition("FAILED")
            self.get_logger().error("land_descent_speed_mps must be finite and positive")
            return
        self.landing_target = [self.position[0], self.position[1]]
        self.last_setpoint_position = [self.landing_target[0], self.landing_target[1], math.nan]
        self.last_setpoint_velocity = [0.0, 0.0, descent_speed]
        self.failure_reason = reason if reason != "goal confirmed" else None
        self.transition("LAND")
        self.last_command_s = -math.inf
        self.get_logger().warning(
            f"Offboard landing started at x={self.landing_target[0]:.3f}, "
            f"y={self.landing_target[1]:.3f}, down_speed={descent_speed:.3f} m/s: {reason}"
        )

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

    def airborne(self):
        return bool(
            self.position is not None
            and self.ground_z is not None
            and self.position[2] < self.ground_z - 0.05
        )

    def odom_fresh(self, now=None):
        now = time.monotonic() if now is None else now
        return self.position is not None and now - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def set_hold(self, position):
        self.last_setpoint_position = [float(value) for value in position]
        self.last_setpoint_velocity = [0.0, 0.0, 0.0]

    def limit_tracking_command(self, command, now):
        """Apply the same indoor speed and acceleration envelope to PX4 commands."""
        speed_limit = float(self.get_parameter("max_horizontal_speed_mps").value)
        acceleration_limit = float(self.get_parameter("max_horizontal_acceleration_mps2").value)
        if speed_limit <= 0.0 or acceleration_limit <= 0.0:
            raise ValueError("horizontal speed and acceleration limits must be positive")
        command = np.asarray(command, dtype=float)
        speed = float(np.linalg.norm(command))
        if speed > speed_limit:
            command *= speed_limit / speed
        previous = self.last_track_command
        previous_time = self.last_track_command_s
        if previous is None or previous_time is None:
            previous = np.zeros(2, dtype=float)
            interval = 1.0 / max(3.0, float(self.get_parameter("publish_hz").value))
        else:
            interval = max(1e-3, now - previous_time)
        delta = command - previous
        max_delta = acceleration_limit * interval
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            command = previous + delta * (max_delta / delta_norm)
        self.last_track_command = command.copy()
        self.last_track_command_s = now
        return command

    def within_target(self, target, tolerance):
        return math.sqrt(sum((self.position[index] - target[index]) ** 2 for index in range(3))) <= tolerance

    def speed(self):
        return math.sqrt(sum(value * value for value in self.velocity))

    def update_dwell(self, now):
        if self.dwell_start_s is None:
            self.dwell_start_s = now

    def transition(self, state):
        if state != "TRACK_QP":
            self.last_track_command = None
            self.last_track_command_s = None
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
        print("Owns PX4 takeoff, verified-waypoint QP tracking, and Offboard landing.")
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
