#!/usr/bin/env python3
"""Visit three poses, look for green_cone, bark if found."""
# 목표지점 1, 2, 3으로 이동하면서 판별. 
# 각도가 흔들리는 이슈로, 목표지점에서 y방향으로 -2 지점에 먼저 간 다음, 서서히 다가오게.
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from std_msgs.msg import Bool, String
from nav_msgs.msg import Path

VISIT_POINTS = [
    (0.21, 15, 1.57),
    (1.21, 15, 1.57),
    (2.21, 15, 1.57),
]
TARGET_LABEL = "cone green"
EVAL_DELAY = 3.0  # seconds to wait after stop


def yaw_to_quaternion(yaw: float):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class FindGreenCone(Node):
    def __init__(self) -> None:
        super().__init__("find_green_cone")
        self.current_pose: Optional[PoseStamped] = None
        self.goal_reached = False
        self.yaw_align_active = False
        self.latest_labels: Optional[str] = None

        self.visit_idx = 0
        self.waiting_for_arrival = False
        self.evaluating = False
        self.eval_start_ns: Optional[int] = None
        self.finished = False
        self.bark_sent = False
        self.approach_phase = True

        qos_goal = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)
        self.speech_pub = self.create_publisher(String, "/robot_dog/speech", 10)
        self.path_pub = self.create_publisher(Path, "/local_path", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(PoseStamped, "/go1_pose", self.pose_cb, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_cb, 10)
        self.create_subscription(Bool, "/yaw_align_active", self.yaw_cb, 10)
        self.create_subscription(String, "/detections/labels", self.labels_cb, 10)

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("find_green_cone started. Waiting for /go1_pose...")

    # Callbacks
    def pose_cb(self, msg):
        self.current_pose = msg
        if not self.waiting_for_arrival and not self.evaluating and not self.finished:
            self.publish_goal()

    def goal_cb(self, msg: Bool):
        self.goal_reached = msg.data

    def yaw_cb(self, msg: Bool):
        self.yaw_align_active = msg.data

    def labels_cb(self, msg: String):
        self.latest_labels = msg.data

    def publish_empty_path(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.cmd_pub.publish(stop)

    # Helpers
    def publish_goal(self):
        if self.visit_idx >= len(VISIT_POINTS):
            return
        x, y, yaw = VISIT_POINTS[self.visit_idx]
        x = float(x)
        y = float(y)
        yaw = float(yaw)
        y_target = y - 2.0 if self.approach_phase else y
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y_target
        goal.pose.position.z = 0.0
        goal.pose.orientation = yaw_to_quaternion(yaw)
        self.goal_pub.publish(goal)
        self.waiting_for_arrival = True
        self.goal_reached = False
        phase = "approach" if self.approach_phase else "target"
        self.get_logger().info(
            f"[Visit {self.visit_idx+1}/3][{phase}] Sent goal to ({x:.2f}, {y_target:.2f}, yaw={yaw:.2f})"
        )

    def condition_met(self) -> bool:
        if not self.latest_labels:
            return False
        labels = [s.strip() for s in self.latest_labels.split(",") if s.strip()]
        return TARGET_LABEL in labels

    def bark_and_shutdown(self):
        if self.bark_sent:
            return
        msg = String()
        msg.data = "bark"
        self.speech_pub.publish(msg)
        self.get_logger().info(f"Detected {TARGET_LABEL}. Barking and shutting down.")
        self.finished = True
        self.bark_sent = True
        if rclpy.ok():
            rclpy.shutdown()

    def control_loop(self):
        if self.finished or self.current_pose is None:
            return

        # waiting for arrival
        if self.waiting_for_arrival:
            if self.goal_reached and not self.yaw_align_active:
                self.waiting_for_arrival = False
                if self.approach_phase:
                    self.approach_phase = False
                    self.publish_goal()
                else:
                    self.publish_empty_path()  # stop robot before evaluating
                    self.evaluating = True
                    self.eval_start_ns = self.get_clock().now().nanoseconds
                    self.get_logger().info(
                        f"Arrived. Waiting {EVAL_DELAY:.1f}s then checking detections..."
                    )
            return

        # evaluating after arrival
        if self.evaluating:
            now_ns = self.get_clock().now().nanoseconds
            elapsed = (now_ns - (self.eval_start_ns or now_ns)) / 1e9
            if elapsed < EVAL_DELAY:
                return
            self.evaluating = False
            if self.condition_met():
                self.bark_and_shutdown()
                return
            # move to next
            self.visit_idx += 1
            self.approach_phase = True
            if self.visit_idx >= len(VISIT_POINTS):
                self.get_logger().info(
                    f"No {TARGET_LABEL} detected after all visits. Shutting down."
                )
                self.finished = True
                rclpy.shutdown()
            else:
                self.publish_goal()
            return

        # initial kick if nothing pending
        if not self.waiting_for_arrival and not self.evaluating:
            self.publish_goal()


def main(args=None):
    rclpy.init(args=args)
    node = FindGreenCone()
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
