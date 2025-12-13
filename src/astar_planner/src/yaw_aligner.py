#!/usr/bin/env python3
"""
Yaw alignment helper node for astar_planner (installed under astar_planner/src).

Sequence:
1) Planner publishes /goal_pose; yaw_aligner stores target yaw.
2) Planner publishes /goal_reached true when position is met.
3) On /goal_reached true, yaw_aligner takes over /cmd_vel (linear.x=0) and
   drives angular.z until yaw error is within tolerance, then stops and
   releases control. If alignment exceeds align_timeout seconds, it aborts
   and releases control to avoid blocking path_tracker.
4) /yaw_align_active (std_msgs/Bool) is published (latched) so consumers
   like path_tracker can pause /cmd_vel while alignment is active.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class YawAligner(Node):
    def __init__(self):
        super().__init__("yaw_aligner")

        # Parameters
        self.declare_parameter("yaw_tolerance", 0.05)       # [rad] widened to finish sooner
        self.declare_parameter("kp_ang", 1.0)               # proportional gain
        self.declare_parameter("max_ang_vel", 0.6)          # [rad/s] slower to reduce overshoot
        self.declare_parameter("slow_zone", 0.4)            # [rad] within this, scale angular speed down
        self.declare_parameter("command_rate", 50.0)        # [Hz] faster checking for quicker align detect

        self.yaw_tol = self.get_parameter("yaw_tolerance").get_parameter_value().double_value
        self.kp_ang = self.get_parameter("kp_ang").get_parameter_value().double_value
        self.max_ang = self.get_parameter("max_ang_vel").get_parameter_value().double_value
        self.slow_zone = self.get_parameter("slow_zone").get_parameter_value().double_value
        self.rate_hz = self.get_parameter("command_rate").get_parameter_value().double_value

        # State
        self.current_pose = None
        self.target_yaw = None
        self.active = False

        # Subscribers
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_reached_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Use default QoS so path_tracker (default subscription) always receives updates
        self.active_pub = self.create_publisher(Bool, "/yaw_align_active", 10)

        # Timer for control loop
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.05
        self.create_timer(period, self.control_loop)

        # Initial state publish
        self.publish_active(False)
        self.get_logger().info("YawAligner ready: waiting for /goal_reached true to align yaw.")

    def pose_cb(self, msg):
        self.current_pose = msg

    def goal_cb(self, msg):
        self.target_yaw = quaternion_to_yaw(msg.pose.orientation)
        self.get_logger().info(f"YawAligner: target yaw set to {self.target_yaw:.3f} rad")

    def goal_reached_cb(self, msg: Bool):
        if msg.data and self.target_yaw is not None:
            # Stop once at goal before starting yaw alignment
            self.send_cmd(0.0)
            self.active = True
            self.publish_active(True)
            self.get_logger().info("YawAligner: goal reached, starting yaw alignment.")

    def publish_active(self, state: bool):
        msg = Bool()
        msg.data = state
        self.active_pub.publish(msg)

    def control_loop(self):
        if not self.active or self.current_pose is None or self.target_yaw is None:
            return

        current_yaw = quaternion_to_yaw(self.current_pose.pose.orientation)
        err = math.atan2(math.sin(self.target_yaw - current_yaw), math.cos(self.target_yaw - current_yaw))

        if abs(err) <= self.yaw_tol:
            # Stop robot before releasing control
            self.send_cmd(0.0)
            self.active = False
            self.publish_active(False)
            self.get_logger().info("YawAligner: yaw aligned, releasing control.")
            return

        # Proportional control with tapered speed near goal to minimize overshoot
        ang = self.kp_ang * err
        if self.slow_zone > 1e-6:
            scale = min(1.0, max(abs(err) / self.slow_zone, 0.1))
            ang *= scale
        ang = max(min(ang, self.max_ang), -self.max_ang)
        self.send_cmd(ang)

    def send_cmd(self, ang_z):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = ang_z
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = YawAligner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_cmd(0.0)
        node.publish_active(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
