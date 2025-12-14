#!/usr/bin/env python3
"""Visit three poses, look for blue_cone, bark if found."""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from std_msgs.msg import Bool, String
from nav_msgs.msg import Path


VISIT_POINTS = [
    (0.21, 14.25, 1.57),
    (1.21, 14.25, 1.57),
    (2.21, 14.25, 1.57),
]
DEFAULT_LABEL = "cone blue"
EVAL_DELAY = 3.0  # seconds to wait after stop

# Robustness parameters
GOAL_BURST_DURATION = 1.0       # seconds: republish fast after sending a new goal
GOAL_KEEPALIVE_PERIOD = 1.0     # seconds: keep publishing goal while navigating (slow)
GOAL_ARMING_DELAY = 0.5         # seconds: ignore goal_reached during this window after sending
ARRIVAL_TIMEOUT = 30.0          # seconds: if "arrival" never happens, re-arm + re-burst
ARRIVAL_DIST_FALLBACK = 0.6     # meters: fallback arrival check using pose vs goal


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class FindCone(Node):
    def __init__(self, target_label: str = DEFAULT_LABEL, node_name: str = "find_blue_cone") -> None:
        super().__init__(node_name)

        # ---- Inputs/state ----
        self.current_pose: Optional[PoseStamped] = None
        self.goal_reached = False
        self.yaw_align_active = False
        self.latest_labels: Optional[str] = None
        self.target_label = target_label

        # ---- Mission state ----
        self.visit_idx = 0
        self.approach_phase = True  # first go to offset point, then actual target

        self.waiting_for_arrival = False
        self.evaluating = False
        self.eval_start_ns: Optional[int] = None

        self.finished = False
        self.bark_sent = False

        # ---- Active goal tracking for robust republish ----
        self.active_goal: Optional[PoseStamped] = None
        self.active_goal_xy: Optional[Tuple[float, float]] = None  # (x, y) for distance fallback

        self.goal_sent_ns: Optional[int] = None
        self.goal_arm_ns: Optional[int] = None
        self.seen_false_after_send = False

        self.last_goal_pub_ns: Optional[int] = None

        # Latched goal publisher (late-joiners still get last goal)
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
        self.get_logger().info("find_blue_cone started. Waiting for /go1_pose...")

    # -----------------------------
    # Callbacks (do NOT publish here)
    # -----------------------------
    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def goal_cb(self, msg: Bool):
        self.goal_reached = msg.data

    def yaw_cb(self, msg: Bool):
        self.yaw_align_active = msg.data

    def labels_cb(self, msg: String):
        self.latest_labels = msg.data

    # -----------------------------
    # Helpers
    # -----------------------------
    def publish_empty_path_and_stop(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.cmd_pub.publish(stop)

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _sec_to_ns(self, s: float) -> int:
        return int(s * 1e9)

    def condition_met(self) -> bool:
        if not self.latest_labels:
            return False
        labels = [s.strip() for s in self.latest_labels.split(",") if s.strip()]
        return self.target_label in labels

    def bark_and_shutdown(self):
        if self.bark_sent:
            return
        msg = String()
        msg.data = "bark"
        self.speech_pub.publish(msg)
        self.get_logger().info(f"Detected {self.target_label}. Barking and shutting down.")
        self.finished = True
        self.bark_sent = True
        rclpy.shutdown()

    def prepare_and_set_active_goal(self):
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

        self.active_goal = goal
        self.active_goal_xy = (x, y_target)

        # Reset navigation bookkeeping
        now_ns = self._now_ns()
        self.goal_sent_ns = now_ns
        self.goal_arm_ns = now_ns + self._sec_to_ns(GOAL_ARMING_DELAY)
        self.seen_false_after_send = False
        self.last_goal_pub_ns = None

        # State flags
        self.waiting_for_arrival = True
        self.goal_reached = False  # local expectation reset

        phase = "approach" if self.approach_phase else "target"
        self.get_logger().info(
            f"[Visit {self.visit_idx+1}/{len(VISIT_POINTS)}][{phase}] "
            f"Active goal set to ({x:.2f}, {y_target:.2f}, yaw={yaw:.2f})"
        )

    def maybe_publish_active_goal(self):
        if self.active_goal is None:
            return

        now_ns = self._now_ns()
        sent_ns = self.goal_sent_ns or now_ns
        elapsed_s = (now_ns - sent_ns) / 1e9

        # Burst phase: publish every control tick (~10Hz)
        if elapsed_s <= GOAL_BURST_DURATION:
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            return

        # Keepalive phase: publish at 1Hz
        if self.last_goal_pub_ns is None or (now_ns - self.last_goal_pub_ns) >= self._sec_to_ns(GOAL_KEEPALIVE_PERIOD):
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            self.last_goal_pub_ns = now_ns

    def arrived(self) -> bool:
        if self.current_pose is None:
            return False

        now_ns = self._now_ns()

        # Arm window: ignore goal_reached for a short time after sending
        if self.goal_arm_ns is not None and now_ns >= self.goal_arm_ns:
            # Track that we have seen "False" after sending (edge-ish guard)
            if not self.goal_reached:
                self.seen_false_after_send = True

        # Primary condition (uses external signals)
        if (
            self.goal_arm_ns is not None
            and now_ns >= self.goal_arm_ns
            and self.seen_false_after_send
            and self.goal_reached
            and (not self.yaw_align_active)
        ):
            return True

        # Fallback: distance-to-goal (if goal_reached is flaky)
        if self.active_goal_xy is not None:
            gx, gy = self.active_goal_xy
            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            dist = math.hypot(gx - cx, gy - cy)
            if dist <= ARRIVAL_DIST_FALLBACK and (not self.yaw_align_active):
                return True

        return False

    def arrival_timed_out(self) -> bool:
        if not self.waiting_for_arrival:
            return False
        if self.goal_sent_ns is None:
            return False
        now_ns = self._now_ns()
        return (now_ns - self.goal_sent_ns) >= self._sec_to_ns(ARRIVAL_TIMEOUT)

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self):
        if self.finished:
            return
        if self.current_pose is None:
            return

        # If we are not navigating/evaluating, ensure we have an active goal
        if (not self.waiting_for_arrival) and (not self.evaluating):
            if self.visit_idx >= len(VISIT_POINTS):
                self.get_logger().info("All visits done. Shutting down.")
                self.finished = True
                rclpy.shutdown()
                return

            # Set next goal (approach or target) as active goal
            self.prepare_and_set_active_goal()

        # If navigating, publish goal robustly and check arrival
        if self.waiting_for_arrival:
            self.maybe_publish_active_goal()

            # Timeout recovery: re-arm + re-burst the same goal
            if self.arrival_timed_out():
                self.get_logger().warn("Arrival timeout. Re-arming and re-bursting the current goal.")
                # Re-arm without changing the goal
                now_ns = self._now_ns()
                self.goal_sent_ns = now_ns
                self.goal_arm_ns = now_ns + self._sec_to_ns(GOAL_ARMING_DELAY)
                self.seen_false_after_send = False
                self.last_goal_pub_ns = None
                return

            if self.arrived():
                self.waiting_for_arrival = False

                if self.approach_phase:
                    # Switch from approach point to actual target point
                    self.approach_phase = False
                    self.prepare_and_set_active_goal()
                    return

                # Stop before evaluating detections
                self.publish_empty_path_and_stop()
                self.evaluating = True
                self.eval_start_ns = self._now_ns()
                self.get_logger().info(f"Arrived. Waiting {EVAL_DELAY}s then checking detections...")
                return

            return

        # Evaluating after arrival
        if self.evaluating:
            now_ns = self._now_ns()
            elapsed = (now_ns - (self.eval_start_ns or now_ns)) / 1e9
            if elapsed < EVAL_DELAY:
                return

            self.evaluating = False

            if self.condition_met():
                self.bark_and_shutdown()
                return

            # Move to next visit
            self.visit_idx += 1
            self.approach_phase = True
            self.active_goal = None
            self.active_goal_xy = None

            if self.visit_idx >= len(VISIT_POINTS):
                self.get_logger().info(f"No {self.target_label} detected after all visits. Shutting down.")
                self.finished = True
                rclpy.shutdown()
                return

            # Next cycle will set goal
            return


def main(args=None):
    rclpy.init(args=args)
    node = FindCone()
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
