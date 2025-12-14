#!/usr/bin/env python3
"""State machine to visit three poses, detect a box, and push it (robust startup/goal handling)."""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32, String


# Visit order (x, y, yaw)
VISIT_POINTS: List[Tuple[float, float, float]] = [
    (4.0, 12.0, 3.13),
    (0.0, 16.0, -1.57),
    (-4.0, 12.0, 0.0),
]

# Push targets corresponding to each visit point
PUSH_POINTS: List[Tuple[float, float, float]] = [
    (0.3, 12.0, 3.13),
    (0.0, 12.3, -1.57),
    (-0.3, 12.0, 0.0),
]

MIN_PATH_POINTS = 60
MAX_PATH_POINTS = 100
PATH_DENSITY = 20  # waypoints per meter

#############
TARGET_LABEL = "suitcase"
MAX_DETECT_DISTANCE = 3.0
EVAL_DURATION = 3.0
PUSH_REACHED_TOL = 0.4
#############

# Robustness parameters (same spirit as FindCone)
GOAL_BURST_DURATION = 1.0        # seconds: republish fast after setting a new visit goal
GOAL_KEEPALIVE_PERIOD = 1.0      # seconds: republish visit goal while navigating (slow)
GOAL_ARMING_DELAY = 0.5          # seconds: ignore goal_reached during this window after sending
ARRIVAL_TIMEOUT = 30.0           # seconds: if arrival never happens, re-arm + re-burst
ARRIVAL_DIST_FALLBACK = 0.7      # meters: fallback arrival check using pose vs goal


def yaw_to_quaternion(yaw: float):
    q = PoseStamped().pose.orientation
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PushTheBox(Node):
    def __init__(self) -> None:
        super().__init__("push_the_box")

        # ---- Inputs ----
        self.current_pose: Optional[PoseStamped] = None
        self.goal_reached = False
        self.yaw_align_active = True
        self.latest_labels: Optional[str] = None
        self.latest_distance: Optional[float] = None

        # ---- Mission state ----
        self.visit_idx = 0
        self.push_done = False
        self.shutdown_requested = False

        self.waiting_for_arrival = False
        self.evaluating = False
        self.eval_start_ns: Optional[int] = None

        # ---- Push tracking ----
        self.push_path_sent = False
        self.push_target: Optional[Tuple[float, float, float]] = None

        # ---- Active VISIT goal tracking (robust republish + anti-stale) ----
        self.active_goal: Optional[PoseStamped] = None
        self.active_goal_xy: Optional[Tuple[float, float]] = None  # (x, y) for distance fallback
        self.goal_sent_ns: Optional[int] = None
        self.goal_arm_ns: Optional[int] = None
        self.seen_false_after_send = False
        self.last_goal_pub_ns: Optional[int] = None

        # Publishers
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", latched_qos)
        self.path_pub = self.create_publisher(Path, "/local_path", 10)

        # Subscribers (callbacks never publish)
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_cb, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_cb, 10)
        self.create_subscription(Bool, "/yaw_align_active", self.yaw_align_cb, 10)
        self.create_subscription(String, "/detections/labels", self.labels_cb, 10)
        self.create_subscription(Float32, "/detections/distance", self.distance_cb, 10)

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("push_the_box node started. Waiting for /go1_pose...")

    # -----------------------------
    # Callbacks (store only)
    # -----------------------------
    def pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def goal_cb(self, msg: Bool) -> None:
        self.goal_reached = msg.data

    def yaw_align_cb(self, msg: Bool) -> None:
        self.yaw_align_active = msg.data

    def labels_cb(self, msg: String) -> None:
        self.latest_labels = msg.data

    def distance_cb(self, msg: Float32) -> None:
        self.latest_distance = float(msg.data)

    # -----------------------------
    # Time helpers
    # -----------------------------
    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _sec_to_ns(self, s: float) -> int:
        return int(s * 1e9)

    # -----------------------------
    # Core helpers
    # -----------------------------
    def publish_empty_path(self) -> None:
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

    def condition_met(self) -> bool:
        if self.latest_labels is None or self.latest_distance is None:
            return False

        labels = [s.strip() for s in self.latest_labels.split(",") if s.strip()]
        has_target = TARGET_LABEL in labels
        dist = float(self.latest_distance)
        dist_ok = (0.0 < dist < MAX_DETECT_DISTANCE)
        return has_target and dist_ok

    # ---- Visit-goal robust publishing ----
    def prepare_active_visit_goal(self) -> None:
        if self.visit_idx >= len(VISIT_POINTS):
            return

        x, y, yaw = VISIT_POINTS[self.visit_idx]
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0
        goal.pose.orientation = yaw_to_quaternion(float(yaw))

        self.active_goal = goal
        self.active_goal_xy = (float(x), float(y))

        now_ns = self._now_ns()
        self.goal_sent_ns = now_ns
        self.goal_arm_ns = now_ns + self._sec_to_ns(GOAL_ARMING_DELAY)
        self.seen_false_after_send = False
        self.last_goal_pub_ns = None

        self.waiting_for_arrival = True
        self.goal_reached = False  # local reset

        self.get_logger().info(
            f"[Visit {self.visit_idx+1}/{len(VISIT_POINTS)}] Active goal set to "
            f"({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
        )

    def maybe_publish_active_goal(self) -> None:
        if self.active_goal is None:
            return

        now_ns = self._now_ns()
        sent_ns = self.goal_sent_ns or now_ns
        elapsed_s = (now_ns - sent_ns) / 1e9

        # Burst phase (fast publish)
        if elapsed_s <= GOAL_BURST_DURATION:
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            return

        # Keepalive (slow publish)
        if self.last_goal_pub_ns is None or (now_ns - self.last_goal_pub_ns) >= self._sec_to_ns(GOAL_KEEPALIVE_PERIOD):
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            self.last_goal_pub_ns = now_ns

    def arrival_timed_out(self) -> bool:
        if not self.waiting_for_arrival:
            return False
        if self.goal_sent_ns is None:
            return False
        return (self._now_ns() - self.goal_sent_ns) >= self._sec_to_ns(ARRIVAL_TIMEOUT)

    def arrived_visit_goal(self) -> bool:
        if self.current_pose is None:
            return False

        now_ns = self._now_ns()

        # Arm window: ignore stale goal_reached right after publish
        if self.goal_arm_ns is not None and now_ns >= self.goal_arm_ns:
            if not self.goal_reached:
                self.seen_false_after_send = True

        # Primary: uses external signals
        if (
            self.goal_arm_ns is not None
            and now_ns >= self.goal_arm_ns
            and self.seen_false_after_send
            and self.goal_reached
            and (not self.yaw_align_active)
        ):
            return True

        # Fallback: distance to active goal
        if self.active_goal_xy is not None:
            gx, gy = self.active_goal_xy
            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            dist = math.hypot(gx - cx, gy - cy)
            if dist <= ARRIVAL_DIST_FALLBACK and (not self.yaw_align_active):
                return True

        return False

    # -----------------------------
    # Push path generation + publish
    # -----------------------------
    def generate_smooth_path(
        self,
        start_pose: PoseStamped,
        target_x: float,
        target_y: float,
        target_yaw: float,
        num_points: int,
    ) -> Path:
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        x0 = start_pose.pose.position.x
        y0 = start_pose.pose.position.y
        z0 = start_pose.pose.position.z
        yaw0 = quaternion_to_yaw(start_pose.pose.orientation)

        x1 = target_x
        y1 = target_y
        z1 = 0.0
        yaw1 = target_yaw

        distance = math.hypot(x1 - x0, y1 - y0)
        control_scale = min(distance * 0.5, 2.0)

        cx0 = x0 + control_scale * math.cos(yaw0)
        cy0 = y0 + control_scale * math.sin(yaw0)
        cx1 = x1 - control_scale * math.cos(yaw1)
        cy1 = y1 - control_scale * math.sin(yaw1)

        tangent_x0 = cx0 - x0
        tangent_y0 = cy0 - y0
        tangent_x1 = x1 - cx1
        tangent_y1 = y1 - cy1

        for i in range(num_points + 1):
            t = i / num_points
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = h00 * x0 + h10 * tangent_x0 + h01 * x1 + h11 * tangent_x1
            pose.pose.position.y = h00 * y0 + h10 * tangent_y0 + h01 * y1 + h11 * tangent_y1
            pose.pose.position.z = z0 + t * (z1 - z0)

            if i < num_points:
                t_next = (i + 1) / num_points
                h00n = 2 * t_next**3 - 3 * t_next**2 + 1
                h10n = t_next**3 - 2 * t_next**2 + t_next
                h01n = -2 * t_next**3 + 3 * t_next**2
                h11n = t_next**3 - t_next**2
                x_next = h00n * x0 + h10n * tangent_x0 + h01n * x1 + h11n * tangent_x1
                y_next = h00n * y0 + h10n * tangent_y0 + h01n * y1 + h11n * tangent_y1
                tangent_yaw = math.atan2(y_next - pose.pose.position.y, x_next - pose.pose.position.x)
            else:
                tangent_yaw = target_yaw

            pose.pose.orientation = yaw_to_quaternion(tangent_yaw)
            path.poses.append(pose)

        return path

    def publish_push_path(self, target: Tuple[float, float, float]) -> None:
        if self.current_pose is None:
            return

        tx, ty, tyaw = target
        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y
        distance = math.hypot(tx - start_x, ty - start_y)

        num_points = max(
            MIN_PATH_POINTS,
            min(MAX_PATH_POINTS, max(int(distance * PATH_DENSITY), MIN_PATH_POINTS)),
        )

        path = self.generate_smooth_path(self.current_pose, tx, ty, tyaw, num_points)
        self.path_pub.publish(path)

        # Switch to push mode: never publish visit goal again
        self.push_path_sent = True
        self.push_target = target

        # Clear visit goal state
        self.waiting_for_arrival = False
        self.evaluating = False
        self.active_goal = None
        self.active_goal_xy = None

        self.get_logger().info(
            f"Published push path to ({tx:.2f}, {ty:.2f}, yaw={tyaw:.2f}) with {len(path.poses)} points."
        )

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self) -> None:
        # Finished mission
        if self.push_done:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                self.publish_empty_path()
                self.get_logger().info("Mission completed. Stopping robot and shutting down node.")
                rclpy.shutdown()
            return

        # Need pose first
        if self.current_pose is None:
            return

        # If push path has been sent, wait until robot reaches push target
        if self.push_path_sent and self.push_target is not None:
            tx, ty, _ = self.push_target
            cur_x = self.current_pose.pose.position.x
            cur_y = self.current_pose.pose.position.y
            dist = math.hypot(tx - cur_x, ty - cur_y)

            if dist <= PUSH_REACHED_TOL:
                self.get_logger().info(f"Push target reached (dist={dist:.3f} m). Shutting down.")
                self.push_done = True
            return

        # Ensure we have an active visit goal if not evaluating
        if (not self.waiting_for_arrival) and (not self.evaluating):
            if self.visit_idx >= len(VISIT_POINTS):
                self.get_logger().info(
                    f"No target '{TARGET_LABEL}' detected within {MAX_DETECT_DISTANCE:.1f} m at any location. Shutting down."
                )
                self.push_done = True
                return
            self.prepare_active_visit_goal()

        # Navigating to visit goal
        if self.waiting_for_arrival:
            self.maybe_publish_active_goal()

            if self.arrival_timed_out():
                self.get_logger().warn("Arrival timeout. Re-arming and re-bursting the current visit goal.")
                now_ns = self._now_ns()
                self.goal_sent_ns = now_ns
                self.goal_arm_ns = now_ns + self._sec_to_ns(GOAL_ARMING_DELAY)
                self.seen_false_after_send = False
                self.last_goal_pub_ns = None
                return

            if self.arrived_visit_goal():
                self.waiting_for_arrival = False
                self.evaluating = True
                self.eval_start_ns = self._now_ns()
                self.get_logger().info(f"Arrived; evaluating detections for {EVAL_DURATION:.1f} seconds...")
            return

        # Evaluating detections at visit point
        if self.evaluating:
            now_ns = self._now_ns()
            elapsed = (now_ns - (self.eval_start_ns or now_ns)) / 1e9

            if self.condition_met():
                target = PUSH_POINTS[self.visit_idx]
                self.publish_push_path(target)
                return

            if elapsed >= EVAL_DURATION:
                self.evaluating = False
                self.visit_idx += 1
                self.active_goal = None
                self.active_goal_xy = None
                # Next loop will set next goal or shut down
            return


def main(args=None):
    rclpy.init(args=args)
    node = PushTheBox()
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
