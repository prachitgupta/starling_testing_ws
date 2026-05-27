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
OWNER_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class MissionTakeoff(Node):
    def __init__(self):
        super().__init__("llm_vision_mission_takeoff")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("offboard_owner_topic", "/llm_vision/offboard_owner")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("takeoff_z", -0.25)
        self.declare_parameter("takeoff_accept_m", 0.08)
        self.declare_parameter("takeoff_settle_s", 2.0)
        self.declare_parameter("pose_timeout_s", 2.0)
        self.declare_parameter("prime_s", 1.5)

        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.offboard_owner_topic = str(self.get_parameter("offboard_owner_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.mission_state_pub = self.create_publisher(String, self.mission_state_topic, 10)
        self.owner_pub = self.create_publisher(String, self.offboard_owner_topic, OWNER_QOS)
        self.odom_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.odom_callback, ODOM_QOS)
        self.owner_sub = self.create_subscription(String, self.offboard_owner_topic, self.owner_callback, OWNER_QOS)

        self.position = None
        self.yaw = math.nan
        self.last_odom_s = 0.0
        self.state = "WAIT_POSE"
        self.state_start_s = time.time()
        self.takeoff_setpoint = None
        self.hold_setpoint = None
        self.timer = self.create_timer(0.05, self.tick)

    def owner_callback(self, msg):
        try:
            owner = json.loads(msg.data).get("owner", "")
        except json.JSONDecodeError:
            owner = msg.data
        if owner in ("trajectory_follower", "trajectory_follower_continuous") and self.state != "HANDED_OFF":
            self.transition("HANDED_OFF")
            self.get_logger().info(f"offboard ownership handed to {owner}; stopping takeoff setpoint stream")

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

    def tick(self):
        if not self.odom_fresh():
            self.publish_mission_state("WAITING_FOR_POSE")
            return

        if self.state == "HANDED_OFF":
            self.publish_mission_state(self.public_state())
            return

        setpoint = self.hold_setpoint or self.takeoff_setpoint or self.position

        if self.state == "WAIT_POSE":
            self.takeoff_setpoint = [self.position[0], self.position[1], float(self.get_parameter("takeoff_z").value)]
            self.transition("PRIME")

        elif self.state == "PRIME" and self.elapsed() >= float(self.get_parameter("prime_s").value):
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.transition("TAKEOFF")

        elif self.state == "TAKEOFF" and self.takeoff_settled():
            self.hold_setpoint = list(self.position)
            self.transition("HOLDING_FOR_PLAN")
            self.get_logger().info("takeoff reached; holding position and publishing mission state")

        self.publish_owner("mission_takeoff")
        self.publish_setpoint(setpoint)
        self.publish_mission_state(self.public_state())

    def public_state(self):
        if self.state == "TAKEOFF":
            return "TAKING_OFF"
        return self.state

    def takeoff_settled(self):
        z_error = abs(self.position[2] - self.takeoff_setpoint[2])
        return (
            z_error <= float(self.get_parameter("takeoff_accept_m").value)
            and self.elapsed() >= float(self.get_parameter("takeoff_settle_s").value)
        )

    def odom_fresh(self):
        return self.position is not None and time.time() - self.last_odom_s <= float(self.get_parameter("pose_timeout_s").value)

    def publish_setpoint(self, position):
        stamp = int(self.get_clock().now().nanoseconds / 1000)

        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = stamp
        setpoint.position = position
        setpoint.velocity = [0.0, 0.0, 0.0]
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

    def publish_owner(self, owner):
        msg = String()
        msg.data = json.dumps({"owner": owner, "timestamp": time.time()})
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
        self.get_logger().info(f"state -> {state}")

    def elapsed(self):
        return time.time() - self.state_start_s


def main():
    rclpy.init()
    node = MissionTakeoff()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
