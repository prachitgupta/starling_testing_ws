#!/usr/bin/env python3
import time

import rclpy
from px4_msgs.msg import VehicleCommand
from rclpy.node import Node


class TakeoffLand(Node):
    def __init__(self):
        super().__init__("voxl_takeoff_land")
        self.declare_parameter("takeoff_alt_m", 0.25)
        self.declare_parameter("hold_s", 30.0)
        self.pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)

    def cmd(self, command, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1, msg.param2, msg.param3, msg.param4 = p1, p2, p3, p4
        msg.param5, msg.param6, msg.param7 = p5, p6, p7
        msg.target_system = msg.source_system = 1
        msg.target_component = msg.source_component = 1
        msg.from_external = True
        self.pub.publish(msg)

    def run(self):
        alt = float(self.get_parameter("takeoff_alt_m").value)
        hold_s = float(self.get_parameter("hold_s").value)
        self.get_logger().info(f"arming, takeoff to {alt:.1f} m, hold {hold_s:.1f} s, land")
        for _ in range(5):
            self.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, p7=alt)
        time.sleep(hold_s)
        self.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)


def main():
    rclpy.init()
    node = TakeoffLand()
    try:
        node.run()
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

