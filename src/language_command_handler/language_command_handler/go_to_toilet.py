#!/usr/bin/env python3
"""Publish a goal to the nearest toilet location (structured + robust)."""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped


TOILET_POINTS = [
    (7.17, -0.5),
    (7.17, -14.5),
    (-7.17, -0.5),
    (-7.17, -14.5),
]
TARGET_YAW = 1.57  # facing +y

# Robustness parameters
GOAL_BURST_DURATION = 1.0        # seconds: republish fast after setting goal
GOAL_KEEPALIVE_PERIOD = 1.0      # seconds: republish while active (slow)
FINISH_AFTER = 3.0               # seconds after sending goal -> shutdown (optional)


def yaw_to_quaternion_z_w(yaw: float) -> Tuple[float, float]:
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class GoToToilet(Node):
    def __init__(self) -> None:
        super().__init__("go_to_toilet")

        # Inputs
        self.latest_pose: Optional[PoseStamped] = None

        # State
        self.state = "WAIT_FOR_POSE"  # WAIT_FOR_POSE -> PUBLISHING -> FINISHED

        # Active goal tracking
        self.active_goal: Optional[PoseStamped] = None
        self.goal_sent_ns: Optional[int] = None
        self.last_goal_pub_ns: Optional[int] = None
        self.finish_ns: Optional[int] = None

        # Subscribers (store only)
        # NOTE: original code had a typo: "/go1_p    ose" -> "/go1_pose"
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_callback, 10)

        # Publisher (latched)
        qos_goal = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)

        # Control loop
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("GoToToilet node started. Waiting for /go1_pose...")

    # -----------------------------
    # Callbacks (store only)
    # -----------------------------
    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    # -----------------------------
    # Time helpers
    # -----------------------------
    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _sec_to_ns(self, s: float) -> int:
        return int(s * 1e9)

    # -----------------------------
    # Goal helpers
    # -----------------------------
    def set_nearest_toilet_goal(self) -> None:
        assert self.latest_pose is not None
        x = self.latest_pose.pose.position.x
        y = self.latest_pose.pose.position.y

        best_pt = min(TOILET_POINTS, key=lambda p: math.hypot(p[0] - x, p[1] - y))

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(best_pt[0])
        goal.pose.position.y = float(best_pt[1])
        goal.pose.position.z = 0.0

        z, w = yaw_to_quaternion_z_w(TARGET_YAW)
        goal.pose.orientation.z = z
        goal.pose.orientation.w = w

        self.active_goal = goal
        now_ns = self._now_ns()
        self.goal_sent_ns = now_ns
        self.last_goal_pub_ns = None
        self.finish_ns = now_ns + self._sec_to_ns(FINISH_AFTER)

        self.get_logger().info(
            f"Active toilet goal set to ({best_pt[0]:.2f}, {best_pt[1]:.2f}) with yaw {TARGET_YAW:.2f}"
        )

    def maybe_publish_goal(self) -> None:
        if self.active_goal is None:
            return

        now_ns = self._now_ns()
        sent_ns = self.goal_sent_ns or now_ns
        elapsed_s = (now_ns - sent_ns) / 1e9

        # Burst
        if elapsed_s <= GOAL_BURST_DURATION:
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            return

        # Keepalive
        if self.last_goal_pub_ns is None or (now_ns - self.last_goal_pub_ns) >= self._sec_to_ns(GOAL_KEEPALIVE_PERIOD):
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            self.last_goal_pub_ns = now_ns

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self) -> None:
        if self.state == "FINISHED":
            return

        if self.latest_pose is None:
            return

        if self.state == "WAIT_FOR_POSE":
            self.set_nearest_toilet_goal()
            self.state = "PUBLISHING"
            return

        if self.state == "PUBLISHING":
            self.maybe_publish_goal()

            # optional auto-finish (this node is "one-shot goal setter")
            if self.finish_ns is not None and self._now_ns() >= self.finish_ns:
                self.get_logger().info("Goal published (burst/keepalive done). Shutting down.")
                self.state = "FINISHED"
                rclpy.shutdown()
            return


def main(args=None):
    rclpy.init(args=args)
    node = GoToToilet()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
