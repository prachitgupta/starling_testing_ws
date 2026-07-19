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
    VehicleStatus,
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
    "-p", "status_timeout_s:=0.30",
    "-p", "transition_timeout_s:=2.0",
    "-p", "debug:=false",
]


class Px4Harness(Node):
    def __init__(self, suffix):
        super().__init__(f"px4_harness_{suffix}")
        self.odom_pub = self.create_publisher(VehicleOdometry, "/fmu/out/vehicle_odometry", ODOM_QOS)
        self.status_pub = self.create_publisher(VehicleStatus, "/fmu/out/vehicle_status", ODOM_QOS)
        self.land_pub = self.create_publisher(VehicleLandDetected, "/fmu/out/vehicle_land_detected", ODOM_QOS)
        self.plan_pub = self.create_publisher(String, "/llm_vision/plan_verified", LATCHED_QOS)
        self.commands = []
        self.setpoints = []
        self.heartbeats = []
        self.mission_states = []
        self.create_subscription(VehicleCommand, "/fmu/in/vehicle_command", self.command_callback, 10)
        self.create_subscription(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.setpoint_callback, 10)
        self.create_subscription(OffboardControlMode, "/fmu/in/offboard_control_mode", self.heartbeat_callback, 10)
        self.create_subscription(String, "/llm_vision/mission_state", self.mission_callback, 10)
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.armed = False
        self.offboard = False
        self.landed = True

    def publish_telemetry(self, odometry=True, status=True):
        if odometry:
            msg = VehicleOdometry()
            msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            msg.position = list(self.position)
            msg.velocity = list(self.velocity)
            msg.q = [1.0, 0.0, 0.0, 0.0]
            self.odom_pub.publish(msg)
        if status:
            msg = VehicleStatus()
            msg.arming_state = (
                VehicleStatus.ARMING_STATE_ARMED if self.armed else VehicleStatus.ARMING_STATE_DISARMED
            )
            msg.nav_state = (
                VehicleStatus.NAVIGATION_STATE_OFFBOARD
                if self.offboard
                else VehicleStatus.NAVIGATION_STATE_MANUAL
            )
            msg.failsafe = False
            self.status_pub.publish(msg)
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
                "workspace": {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.25},
                "obstacles": [],
            }
        )
        self.plan_pub.publish(msg)

    def command_callback(self, msg):
        self.commands.append((time.monotonic(), int(msg.command), float(msg.param1), float(msg.param2)))

    def setpoint_callback(self, msg):
        self.setpoints.append((time.monotonic(), list(msg.position), list(msg.velocity)))

    def heartbeat_callback(self, _msg):
        self.heartbeats.append(time.monotonic())

    def mission_callback(self, msg):
        self.mission_states.append(json.loads(msg.data)["state"])


def spin_until(executor, node, harness, predicate, timeout, odometry=True, status=True):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        harness.publish_telemetry(odometry=odometry, status=status)
        executor.spin_once(timeout_sec=0.01)
        if predicate():
            return
    raise AssertionError(f"timeout waiting in state {node.state}")


def make_nodes(suffix):
    rclpy.init(args=["offboard_harness", *PARAMETERS])
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

        harness.armed = True
        harness.offboard = True
        harness.landed = False
        spin_until(executor, node, harness, lambda: node.state == "TAKEOFF", 0.5)
        harness.position = [0.0, 0.0, -0.25]
        spin_until(executor, node, harness, lambda: node.state == "HOLDING_FOR_PLAN", 1.0)
        spin_until(
            executor,
            node,
            harness,
            lambda: "HOLDING_FOR_PLAN" in harness.mission_states,
            0.3,
        )

        start = {"x": 0.0, "y": 0.0, "z": -0.25}
        goal = {"x": 0.10, "y": 0.0, "z": -0.25}
        harness.publish_plan(True, [start, goal])
        spin_until(executor, node, harness, lambda: node.state == "TRACK_QP", 1.0)
        tracking_index = len(harness.setpoints)
        spin_until(executor, node, harness, lambda: len(harness.setpoints) > tracking_index + 3, 0.3)
        tracking = harness.setpoints[tracking_index:]
        assert any(math.isnan(item[1][0]) and math.isnan(item[1][1]) for item in tracking)
        assert all(math.isfinite(item[1][2]) for item in tracking)
        assert any(all(math.isfinite(value) for value in item[2]) for item in tracking)

        spin_until(executor, node, harness, lambda: node.state == "GOAL_HOLD", 1.0)
        harness.position = [0.10, 0.0, -0.25]
        harness.velocity = [0.0, 0.0, 0.0]
        spin_until(executor, node, harness, lambda: node.state == "LAND", 1.0)
        spin_until(
            executor,
            node,
            harness,
            lambda: any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands),
            0.3,
        )
        assert harness.setpoints and harness.heartbeats

        harness.landed = True
        spin_until(
            executor,
            node,
            harness,
            lambda: any(
                item[1] == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and item[2] == 0.0
                for item in harness.commands
            ),
            0.5,
        )
        harness.armed = False
        harness.offboard = False
        spin_until(executor, node, harness, lambda: node.state == "COMPLETE", 0.5)

        heartbeat_gaps = [later - earlier for earlier, later in zip(harness.heartbeats, harness.heartbeats[1:])]
        assert heartbeat_gaps and max(heartbeat_gaps) < 0.15
    finally:
        destroy_nodes(executor, node, harness)


def test_invalid_plans_land_without_qp():
    cases = [
        ("failed", False, [{"x": 0.0, "y": 0.0, "z": -0.25}, {"x": 0.1, "y": 0.0, "z": -0.25}]),
        ("mismatch", True, [{"x": 0.5, "y": 0.0, "z": -0.25}, {"x": 0.1, "y": 0.0, "z": -0.25}]),
    ]
    for suffix, passed, waypoints in cases:
        executor, node, harness = make_nodes(suffix)
        try:
            spin_until(executor, node, harness, lambda: node.state == "ARM_TAKEOFF", 1.0)
            harness.armed = True
            harness.offboard = True
            harness.landed = False
            harness.position = [0.0, 0.0, -0.25]
            spin_until(executor, node, harness, lambda: node.state == "HOLDING_FOR_PLAN", 1.0)
            harness.publish_plan(passed, waypoints)
            spin_until(executor, node, harness, lambda: node.state == "LAND", 0.5)
            assert node.track_start_s is None
            assert not node.samples
            spin_until(
                executor,
                node,
                harness,
                lambda: any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands),
                0.3,
            )
        finally:
            destroy_nodes(executor, node, harness)


def test_stale_odometry_lands():
    executor, node, harness = make_nodes("stale")
    try:
        spin_until(executor, node, harness, lambda: node.state == "ARM_TAKEOFF", 1.0)
        harness.armed = True
        harness.offboard = True
        harness.landed = False
        harness.position = [0.0, 0.0, -0.10]
        spin_until(executor, node, harness, lambda: node.state == "TAKEOFF", 0.5)
        spin_until(
            executor,
            node,
            harness,
            lambda: node.state == "LAND",
            0.6,
            odometry=False,
            status=True,
        )
        assert node.failure_reason == "odometry timeout"
        spin_until(
            executor,
            node,
            harness,
            lambda: any(item[1] == VehicleCommand.VEHICLE_CMD_NAV_LAND for item in harness.commands),
            0.3,
            odometry=False,
            status=True,
        )
    finally:
        destroy_nodes(executor, node, harness)


if __name__ == "__main__":
    test_complete_mission()
    test_invalid_plans_land_without_qp()
    test_stale_odometry_lands()
    print("unified PX4 offboard ROS harness passed")
