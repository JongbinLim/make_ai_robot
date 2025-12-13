#!/usr/bin/env python3
"""State machine to visit three poses, detect a box, and push it."""

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
    (4.0, 12.0, 3.13),      # 위치 1
    (0.0, 16.0, -1.57),     # 위치 2
    (-4.0, 12.0, 0.0),      # 위치 3
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
TARGET_LABEL = "suitcase"      # 박스로 인식되는 라벨명
MAX_DETECT_DISTANCE = 3.0      # 이 거리 안에 존재해야 함
EVAL_DURATION = 3.0            # 몇 초동안 멈춰서 판단할건지
PUSH_REACHED_TOL = 0.4         # push target 도달 판정 거리 (m)
#############


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

        self.current_pose: Optional[PoseStamped] = None
        self.goal_reached = False
        self.yaw_align_active = True
        self.latest_labels: Optional[str] = None
        self.latest_distance: Optional[float] = None

        self.visit_idx = 0
        self.push_done = False
        self.waiting_for_arrival = False

        self.evaluating = False
        self.eval_start_ns: Optional[int] = None

        self.shutdown_requested = False

        # Push tracking
        self.push_path_sent = False
        self.push_target: Optional[Tuple[float, float, float]] = None

        # Publishers
        # Latched goal publisher so late joiners receive last goal
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", latched_qos)
        self.path_pub = self.create_publisher(Path, "/local_path", 10)

        # Subscribers
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_cb, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_cb, 10)
        self.create_subscription(Bool, "/yaw_align_active", self.yaw_align_cb, 10)
        self.create_subscription(String, "/detections/labels", self.labels_cb, 10)
        self.create_subscription(Float32, "/detections/distance", self.distance_cb, 10)

        # Timer for state checks
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("push_the_box node started. Waiting for /go1_pose...")

    # Subscribers
    def pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg

        # ✅ 중요: push 경로를 이미 보낸 상태라면(또는 평가 중이라면)
        # /go1_pose가 들어올 때마다 방문 goal을 다시 퍼블리시해서 push를 덮어쓰면 안 됨.
        if (
            not self.waiting_for_arrival
            and not self.evaluating
            and not self.push_path_sent
            and not self.push_done
        ):
            self.publish_goal_to_current_visit()

    def goal_cb(self, msg: Bool) -> None:
        self.goal_reached = msg.data

    def yaw_align_cb(self, msg: Bool) -> None:
        self.yaw_align_active = msg.data

    def labels_cb(self, msg: String) -> None:
        self.latest_labels = msg.data

    def distance_cb(self, msg: Float32) -> None:
        self.latest_distance = float(msg.data)

    # Helpers
    def publish_goal_to_current_visit(self) -> None:
        if self.visit_idx >= len(VISIT_POINTS):
            return

        x, y, yaw = VISIT_POINTS[self.visit_idx]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        goal.pose.orientation = yaw_to_quaternion(yaw)

        self.goal_pub.publish(goal)

        self.waiting_for_arrival = True
        self.goal_reached = False

        self.get_logger().info(
            f"[Visit {self.visit_idx+1}/{len(VISIT_POINTS)}] Published goal_pose to "
            f"({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
        )

    def condition_met(self) -> bool:
        if self.latest_labels is None or self.latest_distance is None:
            self.get_logger().info("condition_met: missing labels/distance")
            return False

        labels = [s.strip() for s in self.latest_labels.split(",") if s.strip()]
        has_target = TARGET_LABEL in labels
        dist = float(self.latest_distance)
        dist_ok = (0.0 < dist < MAX_DETECT_DISTANCE)

        self.get_logger().info(
            f"condition_met: labels={labels}, target={TARGET_LABEL}, "
            f"has_target={has_target}, dist={dist:.3f}, dist_ok={dist_ok}"
        )
        return has_target and dist_ok

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

        # ✅ push 상태로 전환 (이후에는 visit goal을 절대 재발행하지 않도록)
        self.push_path_sent = True
        self.push_target = target
        self.evaluating = False
        self.waiting_for_arrival = False  # push 모드에서는 visit 도착 대기 의미 없음

        self.get_logger().info(
            f"Published push path to ({tx:.2f}, {ty:.2f}, yaw={tyaw:.2f}) with {len(path.poses)} points."
        )

    def publish_empty_path(self) -> None:
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

    def control_loop(self) -> None:
        # Mission finished
        if self.push_done:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                # stop robot before shutdown
                self.publish_empty_path()
                self.get_logger().info("Mission completed. Stopping robot and shutting down node.")
                rclpy.shutdown()
            return

        # Need pose
        if self.current_pose is None:
            return

        # If push path has been sent, wait until robot reaches push target
        if self.push_path_sent and self.push_target is not None:
            tx, ty, _ = self.push_target
            cur_x = self.current_pose.pose.position.x
            cur_y = self.current_pose.pose.position.y
            dist = math.hypot(tx - cur_x, ty - cur_y)

            # Optional: debug
            # self.get_logger().info(f"Pushing... dist_to_target={dist:.3f} m")

            if dist <= PUSH_REACHED_TOL:
                self.get_logger().info(f"Push target reached (dist={dist:.3f} m). Shutting down.")
                self.push_done = True
            return

        # Waiting for arrival at visit point
        if self.waiting_for_arrival:
            if self.goal_reached and not self.yaw_align_active:
                self.waiting_for_arrival = False
                self.evaluating = True
                self.eval_start_ns = self.get_clock().now().nanoseconds
                self.get_logger().info(f"Arrived; evaluating detections for {EVAL_DURATION:.1f} seconds...")
        else:
            # Not waiting and not pushing: ensure a goal is published
            if not self.evaluating and self.visit_idx < len(VISIT_POINTS):
                self.publish_goal_to_current_visit()

        # Evaluation window
        if self.evaluating:
            now_ns = self.get_clock().now().nanoseconds
            elapsed = (now_ns - (self.eval_start_ns or now_ns)) / 1e9

            if self.condition_met():
                target = PUSH_POINTS[self.visit_idx]
                self.publish_push_path(target)
                return

            if elapsed >= EVAL_DURATION:
                self.evaluating = False
                self.visit_idx += 1
                if self.visit_idx >= len(VISIT_POINTS):
                    self.get_logger().info(
                        f"No target '{TARGET_LABEL}' detected within {MAX_DETECT_DISTANCE:.1f} m at any location. "
                        "Shutting down."
                    )
                    self.push_done = True
                else:
                    self.publish_goal_to_current_visit()


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
