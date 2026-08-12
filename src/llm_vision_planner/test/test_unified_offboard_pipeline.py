#!/usr/bin/env python3
"""ROS integration harness for the unified PX4 offboard executor."""

import json
import math
import sys
import time
from pathlib import Path

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLandDetected,
    VehicleOdometry,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fine_tuning" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from control_law_executer import ControlLawExecuter, LATCHED_QOS, ODOM_QOS  # noqa: E402


PARAMETERS = [
    "--ros-args",
    "-p", "publish_hz:=50.0",
    "-p", "prime_s:=0.20",
    "-p", "takeoff_settle_s:=0.12",
    "-p", "goal_settle_s:=0.12",
    "-p", "command_retry_s:=0.05",
    "-p", "pose_timeout_s:=0.15",
    "-p", "transition_timeout_s:=2.0",
    "-p", "debug:=false",
]


class Px4Harness(Node):
    def __init__(self, suffix):
        super().__init__(f"px4_harness_{suffix}")
        self.odom_pub = self.create_publisher(VehicleOdometry, "/fmu/out/vehicle_odometry", ODOM_QOS)
        self.land_pub = self.create_publisher(VehicleLandDetected, "/fmu/out/vehicle_land_detected", ODOM_QOS)
        self.plan_pub = self.create_publisher(String, "/llm_vision/plan_verified", LATCHED_QOS)
        self.commands = []
        self.setpoints = []
        self.heartbeats = []
        self.offboard_modes = []
        self.mission_states = []
        self.create_subscription(VehicleCommand, "/fmu/in/vehicle_command", self.command_callback, 10)
        self.create_subscription(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.setpoint_callback, 10)
        self.create_subscription(OffboardControlMode, "/fmu/in/offboard_control_mode", self.heartbeat_callback, 10)
        self.create_subscription(String, "/llm_vision/mission_state", self.mission_callback, 10)
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.landed = True

    def publish_telemetry(self, odometry=True):
        if odometry:
            msg = VehicleOdometry()
            msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            msg.position = list(self.position)
            msg.velocity = list(self.velocity)
            msg.q = [1.0, 0.0, 0.0, 0.0]
            self.odom_pub.publish(msg)
        msg = VehicleLandDetected()
        msg.landed = self.landed
        self.land_pub.publish(msg)

    def publish_plan(self, passed, waypoints):
        msg = String()
        msg.data = json.dumps(
            {
                "plan_id": 7,
                "passed": passed,
                "waypoints": waypoints,
                "workspace": {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.5},
                "obstacles": [],
            }
        )
        self.plan_pub.publish(msg)

    def command_callback(self, msg):
        self.commands.append((time.monotonic(), int(msg.command), float(msg.param1), float(msg.param2)))

    def setpoint_callback(self, msg):
        self.setpoints.append((time.monotonic(), list(msg.position), list(msg.velocity)))

    def heartbeat_callback(self, msg):
        now = time.monotonic()
        self.heartbeats.append(now)
        self.offboard_modes.append((now, bool(msg.position), bool(msg.velocity)))

    def mission_callback(self, msg):
        self.mission_states.append(json.loads(msg.data)["state"])


def spin_until(executor, node, harness, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        harness.publish_telemetry()
        executor.spin_once(timeout_sec=0.01)
        if predicate():
            return
    raise AssertionError(f"timeout waiting in state {node.state}")


def simulate_offboard_landing(executor, node, harness, ground_z=0.0, simulation_dt=0.1, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        harness.publish_telemetry()
        executor.spin_once(timeout_sec=0.01)
        if harness.setpoints:
            _, position, velocity = harness.setpoints[-1]
            if math.isnan(position[2]) and velocity[2] > 0.0:
                descent_velocity = float(velocity[2])
                harness.position[2] = float(
                    min(ground_z, float(harness.position[2]) + descent_velocity * simulation_dt)
                )
                harness.velocity = [0.0, 0.0, descent_velocity]
                if harness.position[2] >= ground_z:
                    harness.position[2] = ground_z
                    harness.velocity = [0.0, 0.0, 0.0]
                    harness.landed = True
        if node.state == "COMPLETE":
            return
    raise AssertionError(f"simulated landing did not complete; position={harness.position}")


def make_nodes(suffix, extra_parameters=None):
    arguments = ["offboard_harness", *PARAMETERS]
    for parameter in extra_parameters or []:
        arguments.extend(["-p", parameter])
    rclpy.init(args=arguments)
    node = ControlLawExecuter()
    harness = Px4Harness(suffix)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(harness)
    return executor, node, harness


def destroy_nodes(executor, node, harness):
    executor.remove_node(node)
    executor.remove_node(harness)
    node.destroy_node()
    harness.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def test_complete_mission():
    executor, node, harness = make_nodes("complete")
    try:
        spin_until(executor, node, harness, lambda: node.state == "PRIME_OFFBOARD", 1.0)
        prime_started = node.state_start_s
        spin_until(executor, node, harness, lambda: time.monotonic() - prime_started >= 0.15, 0.5)
        assert not harness.commands, "arm/mode command was sent before the priming interval"
        spin_until(executor, node, harness, lambda: len(harness.commands) >= 2, 0.5)
        assert harness.commands[0][0] - prime_started >= 0.19
        assert {item[1] for item in harness.commands[:2]} == {
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
        }

        harness.landed = False
        spin_until(executor, node, harness, lambda: node.state == "TAKEOFF", 0.5)
        assert math.isclose(node.takeoff_target[2], -0.5)
        harness.position = [0.0, 0.0, -0.5]
        spin_until(executor, node, harness, lambda: node.state == "HOLDING_FOR_PLAN", 1.0)
        spin_until(
            executor,
            node,
            harness,
            lambda: "HOLDING_FOR_PLAN" in harness.mission_states,
            0.3,
        )

        start = {"x": 0.0, "y": 0.0, "z": -0.5}
        goal = {"x": 0.10, "y": 0.0, "z": -0.5}
        harness.publish_plan(True, [start, goal])
        spin_until(executor, node, harness, lambda: node.state == "TRACK_QP", 1.0)
        tracking_index = len(harness.setpoints)
        spin_until(executor, node, harness, lambda: len(harness.setpoints) > tracking_index + 3, 0.3)
        tracking = harness.setpoints[tracking_index:]
        assert any(math.isnan(item[1][0]) and math.isnan(item[1][1]) for item in tracking)
        assert all(math.isfinite(item[1][2]) for item in tracking)
        assert any(all(math.isfinite(value) for value in item[2]) for item in tracking)
        horizontal_commands = [item for item in tracking if all(math.isfinite(value) for value in item[2][:2])]
        assert max(math.hypot(item[2][0], item[2][1]) for item in horizontal_commands) <= 0.5 + 1e-6

        node.last_track_command = None
        node.last_track_command_s = None
        first = node.limit_tracking_command([2.0, 0.0], 100.0)
        second = node.limit_tracking_command([2.0, 0.0], 100.02)
        assert math.hypot(*first) <= 0.5 + 1e-6
        assert math.hypot(*second) <= 0.5 + 1e-6
        assert math.hypot(*(second - first)) <= 0.5 * 0.02 + 1e-6
        node.last_track_command = None
        node.last_track_command_s = None

        spin_until(executor, node, harness, lambda: node.state == "GOAL_HOLD", 2.0)
        harness.position = [0.10, 0.0, -0.5]
        harness.velocity = [0.0, 0.0, 0.0]
        spin_until(executor, node, harness, lambda: node.state == "LAND", 1.0)
        landing_started = node.state_start_s
        spin_until(
            executor,
            node,
            harness,
            lambda: len([item for item in harness.setpoints if item[0] >= landing_started]) >= 6,
            0.3,
        )
        landing_setpoints = harness.setpoints[-4:]
        landing_modes = harness.offboard_modes[-4:]
        descent_speed = float(node.get_parameter("land_descent_speed_mps").value)
        assert all(
            math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(node.landing_target, [0.10, 0.0])
        )
        assert landing_modes and all(position and velocity for _, position, velocity in landing_modes)
        assert all(
            math.isclose(position[0], 0.10, abs_tol=1e-6)
            and math.isclose(position[1], 0.0, abs_tol=1e-6)
            and math.isnan(position[2])
            and math.isclose(velocity[0], 0.0, abs_tol=1e-6)
            and math.isclose(velocity[1], 0.0, abs_tol=1e-6)
            and math.isclose(velocity[2], descent_speed, abs_tol=1e-6)
            for _, position, velocity in landing_setpoints
        )
        assert not any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands)

        simulate_offboard_landing(executor, node, harness)
        spin_until(
            executor,
            node,
            harness,
            lambda: any(
                item[1] == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and item[2] == 0.0
                for item in harness.commands
            ),
            timeout=0.5,
        )
        assert math.isclose(harness.position[0], 0.10, abs_tol=1e-6)
        assert math.isclose(harness.position[1], 0.0, abs_tol=1e-6)
        assert math.isclose(harness.position[2], 0.0, abs_tol=1e-6)
        assert any(
            item[1] == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and item[2] == 0.0
            for item in harness.commands
        )

        heartbeat_gaps = [later - earlier for earlier, later in zip(harness.heartbeats, harness.heartbeats[1:])]
        assert heartbeat_gaps and max(heartbeat_gaps) < 0.15
    finally:
        destroy_nodes(executor, node, harness)


def test_invalid_plans_land_without_qp():
    cases = [
        ("failed", False, [{"x": 0.0, "y": 0.0, "z": -0.5}, {"x": 0.1, "y": 0.0, "z": -0.5}]),
        ("mismatch", True, [{"x": 0.5, "y": 0.0, "z": -0.5}, {"x": 0.1, "y": 0.0, "z": -0.5}]),
    ]
    for suffix, passed, waypoints in cases:
        executor, node, harness = make_nodes(suffix)
        try:
            spin_until(executor, node, harness, lambda: node.state == "ARM_TAKEOFF", 1.0)
            harness.landed = False
            harness.position = [0.0, 0.0, -0.5]
            spin_until(executor, node, harness, lambda: node.state == "HOLDING_FOR_PLAN", 1.0)
            harness.publish_plan(passed, waypoints)
            spin_until(executor, node, harness, lambda: node.state == "LAND", 0.5)
            landing_started = node.state_start_s
            assert node.track_start_s is None
            assert not node.samples
            spin_until(
                executor,
                node,
                harness,
                lambda: any(item[0] >= landing_started for item in harness.setpoints),
                0.3,
            )
            assert all(math.isclose(value, 0.0) for value in node.landing_target)
            assert not any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands)
        finally:
            destroy_nodes(executor, node, harness)


def test_successful_mission_holds_when_landing_disabled():
    executor, node, harness = make_nodes("hold", ["land_after_complete:=false"])
    try:
        spin_until(executor, node, harness, lambda: node.state == "ARM_TAKEOFF", 1.0)
        harness.landed = False
        harness.position = [0.0, 0.0, -0.5]
        spin_until(executor, node, harness, lambda: node.state == "HOLDING_FOR_PLAN", 1.0)

        start = {"x": 0.0, "y": 0.0, "z": -0.5}
        goal = {"x": 0.10, "y": 0.0, "z": -0.5}
        harness.publish_plan(True, [start, goal])
        spin_until(executor, node, harness, lambda: node.state == "GOAL_HOLD", 2.0)
        harness.position = [0.10, 0.0, -0.5]
        harness.velocity = [0.0, 0.0, 0.0]
        spin_until(executor, node, harness, lambda: node.state == "HOLDING_AT_GOAL", 1.0)

        setpoint_index = len(harness.setpoints)
        spin_until(executor, node, harness, lambda: len(harness.setpoints) >= setpoint_index + 5, 0.4)
        assert node.state == "HOLDING_AT_GOAL"
        assert "HOLDING_AT_GOAL" in harness.mission_states
        assert all(
            all(math.isclose(actual, expected, abs_tol=1e-6) for actual, expected in zip(position, [0.10, 0.0, -0.5]))
            and all(math.isclose(value, 0.0, abs_tol=1e-6) for value in velocity)
            for _, position, velocity in harness.setpoints[-5:]
        )
        assert not any(
            item[1] in (
                VehicleCommand.VEHICLE_CMD_NAV_LAND,
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            )
            and item[2] == 0.0
            for item in harness.commands
        )
    finally:
        destroy_nodes(executor, node, harness)


def test_stale_odometry_lands():
    executor, node, harness = make_nodes("stale")
    try:
        spin_until(executor, node, harness, lambda: node.state == "ARM_TAKEOFF", 1.0)
        harness.landed = False
        harness.position = [0.0, 0.0, -0.10]
        spin_until(executor, node, harness, lambda: node.state == "TAKEOFF", 0.5)
        spin_until(executor, node, harness, lambda: node.position[2] < -0.05, 0.3)
        node.last_odom_s = time.monotonic() - float(node.get_parameter("pose_timeout_s").value) - 0.01
        node.tick()
        assert node.state == "LAND"
        assert node.failure_reason == "odometry timeout"
        assert all(math.isclose(value, 0.0) for value in node.landing_target)
        assert math.isnan(node.last_setpoint_position[2])
        assert node.last_setpoint_velocity[2] > 0.0
        assert not any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands)
    finally:
        destroy_nodes(executor, node, harness)


if __name__ == "__main__":
    test_complete_mission()
    test_invalid_plans_land_without_qp()
    test_successful_mission_holds_when_landing_disabled()
    test_stale_odometry_lands()
    print("unified PX4 offboard ROS harness passed")
