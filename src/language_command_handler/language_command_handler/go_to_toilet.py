#!/usr/bin/env python3
"""Publish a goal to the nearest toilet location."""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


TOILET_POINTS = [
    (7.17, -0.5),
    (7.17, -14.5),
    (-7.17, -0.5),
    (-7.17, -14.5),
]
TARGET_YAW = 1.57  # facing +y


class GoToToilet(Node):
    def __init__(self) -> None:
        super().__init__("go_to_toilet")
        self.pose_sub = self.create_subscription(
            PoseStamped, "/go1_pose", self.pose_callback, 10
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.latest_pose = None
        self.goal_sent = False
        self.get_logger().info("GoToToilet node started. Waiting for /go1_pose...")

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        if not self.goal_sent:
            self.publish_nearest_goal()

    def publish_nearest_goal(self) -> None:
        if self.latest_pose is None:
            return

        x = self.latest_pose.pose.position.x
        y = self.latest_pose.pose.position.y

        best_pt = min(TOILET_POINTS, key=lambda p: math.hypot(p[0] - x, p[1] - y))
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = best_pt[0]
        goal.pose.position.y = best_pt[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(TARGET_YAW / 2.0)
        goal.pose.orientation.w = math.cos(TARGET_YAW / 2.0)

        self.goal_pub.publish(goal)
        self.goal_sent = True
        self.get_logger().info(
            f"Published toilet goal to ({best_pt[0]:.2f}, {best_pt[1]:.2f}) with yaw {TARGET_YAW:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoToToilet()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
