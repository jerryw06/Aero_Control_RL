#!/usr/bin/env python3
"""
Minimal script to automatically switch PX4 into OFFBOARD mode and arm it.
Keeps publishing offboard heartbeats so the vehicle stays active.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy
)
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand


class AutoOffboardArm(Node):
    def __init__(self):
        super().__init__('auto_offboard_arm_minimal')

        # QoS setup
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        # Publishers
        self.pub_offboard_mode = self.create_publisher(
            OffboardControlMode, 'fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj = self.create_publisher(
            TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, 'fmu/in/vehicle_command', qos_pub)

        # Timer (50 Hz → 0.02 s)
        self.timer = self.create_timer(0.001, self.cmdloop_callback)
        self.setpoint_counter = 0
        self.mode_set = False
        self.armed = False

        self.get_logger().info('Node started — waiting before OFFBOARD switch...')
        time.sleep(2.0)

    # Helper to send vehicle commands
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    def cmdloop_callback(self):
        # Always send offboard heartbeat
        offboard = OffboardControlMode()
        offboard.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard.position = True
        self.pub_offboard_mode.publish(offboard)

        # Dummy trajectory to keep PX4 happy
        traj = TrajectorySetpoint()
        traj.timestamp = offboard.timestamp
        traj.position[0] = 0.0
        traj.position[1] = 0.0
        traj.position[2] = -1.0  # stay at 1 m altitude target
        self.pub_traj.publish(traj)

        self.setpoint_counter += 1

        # After a few heartbeats, request OFFBOARD mode
        if self.setpoint_counter == 10 and not self.mode_set:
            self.get_logger().info('Requesting OFFBOARD mode...')
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1,  # base mode
                param2=6   # custom main mode: OFFBOARD
            )
            self.mode_set = True

        # Then arm
        if self.setpoint_counter == 15 and not self.armed:
            self.get_logger().info('Arming vehicle...')
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1  # 1 = arm
            )
            self.armed = True


def main(args=None):
    rclpy.init(args=args)
    node = AutoOffboardArm()
    try:
        rclpy.spin(node)  # stays alive until Ctrl +C
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down manually.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
