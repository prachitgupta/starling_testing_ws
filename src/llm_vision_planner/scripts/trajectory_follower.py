#!/usr/bin/env python3
import json
import math
import time

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def scale(v, s):
    return [v[i] * s for i in range(3)]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def unit(v):
    length = norm(v)
    if length < 1e-6:
        return [0.0, 0.0, 0.0]
    return [x / length for x in v]


def bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    return [
        u**3 * p0[i] + 3.0 * u**2 * t * p1[i] + 3.0 * u * t**2 * p2[i] + t**3 * p3[i]
        for i in range(3)
    ]


def bezier_velocity(p0, p1, p2, p3, t, duration_s):
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


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__("llm_vision_trajectory_follower")
        self.declare_parameter("plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("accept_m", 0.1)
        self.declare_parameter("start_accept_m", 0.75)
        self.declare_parameter("pose_timeout_s", 0.5)
        self.declare_parameter("prime_s", 1.5)
        self.declare_parameter("takeoff_z", -0.45)
        self.declare_parameter("takeoff_accept_m", 0.10)
        self.declare_parameter("takeoff_settle_s", 1.0)
        self.declare_parameter("auto_arm", True)
        self.declare_parameter("land_after_mission", True)
        self.declare_parameter("hold_after_mission", False)
        self.declare_parameter("min_segment_duration_s", 0.8)

        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.mission_state_pub = self.create_publisher(String, self.mission_state_topic, 10)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.plan_sub = self.create_subscription(String, self.plan_topic, self.plan_callback, 10)

        self.position = None
        self.yaw = math.nan
        self.last_odom_s = 0.0
        self.state = "WAIT_POSE"
        self.state_start_s = time.time()
        self.hold_setpoint = None
        self.waypoints = []
        self.segment_index = 0
        self.segment_start_s = 0.0
        self.timer = self.create_timer(0.05, self.tick)

    def odom_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        self.position = [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])]
        self.yaw = self.quat_to_yaw(msg.q[1], msg.q[2], msg.q[3], msg.q[0])
        self.last_odom_s = time.time()

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def plan_callback(self, msg):
        if self.state != "WAIT_PLAN":
            self.get_logger().warn("ignoring verified plan because follower is not waiting for one", throttle_duration_sec=3.0)
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"failed to parse verified plan JSON: {exc}")
            return

        if not payload.get("passed", False):
            self.get_logger().warn("ignoring verified plan because passed=false")
            return

        waypoints = []
        for waypoint in payload.get("waypoints", []):
            if all(k in waypoint for k in ("x", "y", "z")):
                waypoints.append([float(waypoint["x"]), float(waypoint["y"]), float(waypoint["z"])])

        if not waypoints:
            self.get_logger().warn("ignoring verified plan with no usable waypoints")
            return

        start_error = norm(sub(waypoints[0], self.position))
        if start_error > float(self.get_parameter("start_accept_m").value):
            self.get_logger().error(
                f"rejecting verified plan: first waypoint is {start_error:.2f} m away; "
                f"limit is {float(self.get_parameter('start_accept_m').value):.2f} m"
            )
            return

        self.waypoints = waypoints
        self.segment_index = 0
        self.segment_start_s = time.time()
        self.transition("TRACK")
        self.get_logger().info(f"latched verified trajectory with {len(self.waypoints)} waypoints")

    def tick(self):
        if not self.odom_fresh():
            self.get_logger().warn(f"waiting for fresh PX4 odometry on {self.pose_topic}", throttle_duration_sec=2.0)
            self.publish_mission_state("WAITING_FOR_POSE")
            return

        position_setpoint = self.hold_setpoint or self.position
        velocity_setpoint = [math.nan, math.nan, math.nan]

        if self.state == "WAIT_POSE":
            self.transition("PRIME")

        elif self.state == "PRIME" and self.elapsed() >= float(self.get_parameter("prime_s").value):
            if bool(self.get_parameter("auto_arm").value):
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.transition("TAKEOFF")

        elif self.state == "TAKEOFF" and self.takeoff_settled():
            self.transition("WAIT_PLAN")
            self.get_logger().info(f"waiting for passed=true trajectory on {self.plan_topic}")

        elif self.state == "TRACK":
            position_setpoint, velocity_setpoint = self.segment_setpoint()

        elif self.state == "HOLD" and self.waypoints:
            position_setpoint = self.waypoints[-1]

        self.publish_setpoint(position_setpoint, velocity_setpoint)
        self.publish_mission_state(self.public_state())

    def public_state(self):
        if self.state == "WAIT_PLAN":
            return "HOLDING_FOR_PLAN"
        if self.state == "TRACK":
            return "TRACKING"
        if self.state == "TAKEOFF":
            return "TAKING_OFF"
        if self.state == "LAND":
            return "LANDING"
        return self.state

    def segment_setpoint(self):
        if self.segment_index >= len(self.waypoints) - 1:
            return self.finish_mission()

        p0 = self.waypoints[self.segment_index]
        p3 = self.waypoints[self.segment_index + 1]
        duration_s = self.segment_duration(p0, p3)
        t = min(1.0, (time.time() - self.segment_start_s) / duration_s)
        v0 = self.tangent_velocity(self.segment_index)
        v1 = self.tangent_velocity(self.segment_index + 1)
        p1 = add(p0, scale(v0, duration_s / 3.0))
        p2 = sub(p3, scale(v1, duration_s / 3.0))
        position_setpoint = bezier(p0, p1, p2, p3, t)
        velocity_setpoint = bezier_velocity(p0, p1, p2, p3, t, duration_s)

        if t >= 1.0 or norm(sub(self.position, p3)) <= float(self.get_parameter("accept_m").value):
            self.segment_index += 1
            self.segment_start_s = time.time()
            if self.segment_index >= len(self.waypoints) - 1:
                return self.finish_mission()

        return position_setpoint, velocity_setpoint

    def tangent_velocity(self, index):
        speed = float(self.get_parameter("speed").value)
        if len(self.waypoints) < 2:
            return [0.0, 0.0, 0.0]
        if index == 0:
            direction = sub(self.waypoints[1], self.waypoints[0])
        elif index == len(self.waypoints) - 1:
            direction = sub(self.waypoints[-1], self.waypoints[-2])
        else:
            direction = sub(self.waypoints[index + 1], self.waypoints[index - 1])
        return scale(unit(direction), speed)

    def segment_duration(self, p0, p1):
        speed = max(0.05, float(self.get_parameter("speed").value))
        minimum = float(self.get_parameter("min_segment_duration_s").value)
        return max(minimum, norm(sub(p1, p0)) / speed)

    def finish_mission(self):
        final = self.waypoints[-1]
        self.hold_setpoint = final
        if bool(self.get_parameter("hold_after_mission").value) or not bool(self.get_parameter("land_after_mission").value):
            self.transition("HOLD")
            self.get_logger().info("final waypoint reached; holding")
        else:
            self.transition("LAND")
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("final waypoint reached; landing")
        return final, [0.0, 0.0, 0.0]

    def takeoff_settled(self):
        if self.hold_setpoint is None:
            return False
        z_error = abs(self.position[2] - self.hold_setpoint[2])
        return (
            z_error <= float(self.get_parameter("takeoff_accept_m").value)
            and self.elapsed() >= float(self.get_parameter("takeoff_settle_s").value)
        )

    def odom_fresh(self):
        return self.position is not None and time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

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

    def publish_mission_state(self, state):
        msg = String()
        msg.data = json.dumps(
            {
                "state": state,
                "position": {
                    "x": self.position[0] if self.position else None,
                    "y": self.position[1] if self.position else None,
                    "z": self.position[2] if self.position else None,
                },
                "heading_deg": math.degrees(self.yaw) if math.isfinite(self.yaw) else None,
                "timestamp": time.time(),
            }
        )
        self.mission_state_pub.publish(msg)

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
        if state == "PRIME" and self.position is not None:
            self.hold_setpoint = [
                self.position[0],
                self.position[1],
                float(self.get_parameter("takeoff_z").value),
            ]
        self.state = state
        self.state_start_s = time.time()
        self.get_logger().info(f"state -> {state}")

    def elapsed(self):
        return time.time() - self.state_start_s


def main():
    rclpy.init()
    node = TrajectoryFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
