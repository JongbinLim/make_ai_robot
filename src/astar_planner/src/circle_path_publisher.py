#!/usr/bin/env python3
"""Generate autonomous polygonal paths using the same logic as move_go1.py."""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path


MIN_PATH_POINTS = 60
MAX_PATH_POINTS = 100
PATH_DENSITY = 20


def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw (rad) from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle (rad) into a quaternion."""
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


class CirclePathPublisher(Node):
    """Publish polygonal paths built with the MoveGo1 Hermite spline logic."""

    def __init__(self) -> None:
        super().__init__("circle_path_publisher")

        self.current_pose: Optional[PoseStamped] = None
        self.targets: List[Tuple[float, float, float]] = []  # (x, y, yaw)
        self.target_index: int = 0
        self.current_target: Optional[Tuple[float, float, float]] = None
        self.path_active: bool = False
        self.sequence_finished: bool = False
        self.last_path_publish_ns: Optional[int] = None

        # Parameters
        self.declare_parameter("d", 1.5)
        self.declare_parameter("distance_threshold", 0.6)
        self.declare_parameter("replan_interval", 1.5)

        self.d = self.get_parameter("d").get_parameter_value().double_value
        self.distance_threshold = (
            self.get_parameter("distance_threshold").get_parameter_value().double_value
        )
        self.replan_interval = (
            self.get_parameter("replan_interval").get_parameter_value().double_value
        )
        self.replan_interval_ns = int(self.replan_interval * 1e9)

        # ROS interfaces
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_cb, 10)
        self.path_pub = self.create_publisher(Path, "/local_path", 10)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            "CirclePathPublisher running with internal path generation."
        )

    # ------------------------------------------------------------------
    # Callbacks & state helpers
    # ------------------------------------------------------------------

    def pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg

        if not self.targets and not self.sequence_finished:
            self.init_targets_from_current_pose()

    def init_targets_from_current_pose(self) -> None:
        if self.current_pose is None:
            return

        d = self.d
        x0 = self.current_pose.pose.position.x
        y0 = self.current_pose.pose.position.y
        yaw0 = quaternion_to_yaw(self.current_pose.pose.orientation)

        # build hexagon vertices around a center that sits in front of the robot
        center_x = d
        center_y = 0.0
        radius = d
        local_hex_points: List[Tuple[float, float]] = []
        for k in range(6):
            angle = -math.pi / 2.0 + k * (math.pi / 3.0)  # 60 deg steps
            xr = center_x + radius * math.cos(angle)
            yr = center_y + radius * math.sin(angle)
            local_hex_points.append((xr, yr))
        # return to start pose to complete the loop
        local_hex_points.append((0.0, 0.0))

        world_points: List[Tuple[float, float]] = []
        for xr, yr in local_hex_points:
            x_w = x0 + math.cos(yaw0) * xr - math.sin(yaw0) * yr
            y_w = y0 + math.sin(yaw0) * xr + math.cos(yaw0) * yr
            world_points.append((x_w, y_w))

        yaw_targets: List[float] = []
        for idx in range(len(world_points)):
            if idx < len(world_points) - 1:
                dx = world_points[idx + 1][0] - world_points[idx][0]
                dy = world_points[idx + 1][1] - world_points[idx][1]
                yaw_targets.append(math.atan2(dy, dx))
            else:
                yaw_targets.append(yaw0)

        self.targets = []
        for (x_w, y_w), yaw_t in zip(world_points, yaw_targets):
            self.targets.append((x_w, y_w, yaw_t))

        self.target_index = 0
        self.current_target = None
        self.path_active = False
        self.sequence_finished = False

        self.get_logger().info(
            "Square (circle) targets initialized. Publishing first path soon."
        )

    # ------------------------------------------------------------------
    # Path generation (lifted from move_go1.py)
    # ------------------------------------------------------------------

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

        distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
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

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            pose_stamped.pose.position.x = (
                h00 * x0 + h10 * tangent_x0 + h01 * x1 + h11 * tangent_x1
            )
            pose_stamped.pose.position.y = (
                h00 * y0 + h10 * tangent_y0 + h01 * y1 + h11 * tangent_y1
            )
            pose_stamped.pose.position.z = z0 + t * (z1 - z0)

            if i < num_points:
                t_next = (i + 1) / num_points
                h00_next = 2 * t_next**3 - 3 * t_next**2 + 1
                h10_next = t_next**3 - 2 * t_next**2 + t_next
                h01_next = -2 * t_next**3 + 3 * t_next**2
                h11_next = t_next**3 - t_next**2

                x_next = (
                    h00_next * x0
                    + h10_next * tangent_x0
                    + h01_next * x1
                    + h11_next * tangent_x1
                )
                y_next = (
                    h00_next * y0
                    + h10_next * tangent_y0
                    + h01_next * y1
                    + h11_next * tangent_y1
                )

                dx = x_next - pose_stamped.pose.position.x
                dy = y_next - pose_stamped.pose.position.y
                tangent_yaw = math.atan2(dy, dx)
            else:
                tangent_yaw = yaw1

            pose_stamped.pose.orientation = yaw_to_quaternion(tangent_yaw)
            path.poses.append(pose_stamped)

        return path

    def publish_path_to_current_target(self) -> None:
        if self.current_pose is None:
            return
        if self.target_index >= len(self.targets):
            self.sequence_finished = True
            return

        target_x, target_y, target_yaw = self.targets[self.target_index]
        x_start = self.current_pose.pose.position.x
        y_start = self.current_pose.pose.position.y
        distance = math.hypot(target_x - x_start, target_y - y_start)
        num_points = max(
            MIN_PATH_POINTS,
            min(MAX_PATH_POINTS, max(int(distance * PATH_DENSITY), MIN_PATH_POINTS)),
        )

        path = self.generate_smooth_path(
            self.current_pose, target_x, target_y, target_yaw, num_points
        )
        self.path_pub.publish(path)
        self.path_active = True
        self.current_target = (target_x, target_y, target_yaw)
        self.last_path_publish_ns = self.get_clock().now().nanoseconds

        self.get_logger().info(
            f"Published path {self.target_index + 1}/{len(self.targets)} "
            f"with {len(path.poses)} points."
        )

    def republish_current_path(self) -> None:
        """Replan from the live pose to the same target to keep the robot moving."""
        if self.current_pose is None or self.current_target is None:
            return

        target_x, target_y, target_yaw = self.current_target
        x_start = self.current_pose.pose.position.x
        y_start = self.current_pose.pose.position.y
        distance = math.hypot(target_x - x_start, target_y - y_start)
        num_points = max(
            MIN_PATH_POINTS,
            min(MAX_PATH_POINTS, max(int(distance * PATH_DENSITY), MIN_PATH_POINTS)),
        )

        path = self.generate_smooth_path(
            self.current_pose, target_x, target_y, target_yaw, num_points
        )
        self.path_pub.publish(path)
        self.last_path_publish_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f"Re-published current leg (remaining distance {distance:.2f} m)."
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def control_loop(self) -> None:
        if self.sequence_finished or self.current_pose is None:
            return

        if not self.targets:
            return

        if not self.path_active:
            self.publish_path_to_current_target()
            return

        if self.current_target is None:
            return

        target_x, target_y, target_yaw = self.current_target
        cur_x = self.current_pose.pose.position.x
        cur_y = self.current_pose.pose.position.y

        distance_err = math.hypot(target_x - cur_x, target_y - cur_y)
        if distance_err > self.distance_threshold:
            now_ns = self.get_clock().now().nanoseconds
            if (
                self.last_path_publish_ns is None
                or now_ns - self.last_path_publish_ns >= self.replan_interval_ns
            ):
                self.republish_current_path()

        if distance_err <= self.distance_threshold:
            self.get_logger().info(
                f"Target {self.target_index + 1}/{len(self.targets)} reached."
            )
            self.target_index += 1
            self.current_target = None
            self.path_active = False

            if self.target_index >= len(self.targets):
                self.sequence_finished = True
                self.get_logger().info(
                    "All targets completed. Node will remain idle until restarted."
                )


def main(args=None):
    rclpy.init(args=args)
    node = CirclePathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
