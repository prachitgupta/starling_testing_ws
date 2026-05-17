#!/usr/bin/env python3
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
from rclpy.node import Node


class TrajectoryWaypoint(Node):
    def __init__(self):
        super().__init__("voxl_fmu_trajectory_waypoint")
        self.declare_parameter("x", 0.5)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", -0.5)
        self.declare_parameter("arm_and_offboard", False)
        self.declare_parameter("move_after_s", 3.0)
        self.offboard = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.setpoint = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.command = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_position_callback, 10)
        self.count = 0
        self.local_position = None
        self.hold_xy = None
        self.hold_heading = float("nan")
        self.create_timer(0.05, self.tick)

    def local_position_callback(self, msg):
        if not msg.xy_valid or not msg.z_valid:
            return
        self.local_position = msg
        if self.hold_xy is None:
            self.hold_xy = (float(msg.x), float(msg.y))
            if msg.heading_good_for_control:
                self.hold_heading = float(msg.heading)
            self.get_logger().info(
                f"holding current XY before move: x={self.hold_xy[0]:.2f}, y={self.hold_xy[1]:.2f}")

    def vehicle_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = msg.source_system = 1
        msg.target_component = msg.source_component = 1
        msg.from_external = True
        self.command.publish(msg)

    def tick(self):
        if self.local_position is None or self.hold_xy is None:
            self.get_logger().warn("waiting for /fmu/out/vehicle_local_position", throttle_duration_sec=2.0)
            return

        now = int(self.get_clock().now().nanoseconds / 1000)
        hb = OffboardControlMode()
        hb.timestamp = now
        hb.position = True
        self.offboard.publish(hb)

        move_after_ticks = max(20, int(float(self.get_parameter("move_after_s").value) / 0.05))
        if self.count < move_after_ticks:
            x, y = self.hold_xy
            z = float(self.get_parameter("z").value)
        else:
            x = float(self.get_parameter("x").value)
            y = float(self.get_parameter("y").value)
            z = float(self.get_parameter("z").value)
            if self.count == move_after_ticks:
                self.get_logger().info(f"moving to waypoint: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        sp = TrajectorySetpoint()
        sp.timestamp = now
        sp.position = [x, y, z]
        sp.yaw = self.hold_heading
        self.setpoint.publish(sp)

        if self.count == 20 and bool(self.get_parameter("arm_and_offboard").value):
            self.get_logger().info("requesting Offboard mode and arm")
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.count += 1


def main():
    rclpy.init()
    node = TrajectoryWaypoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

