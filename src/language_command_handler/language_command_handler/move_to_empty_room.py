#!/usr/bin/env python3
"""
Navigate to the nearest room.
If a stop sign is detected strictly near the target room, switch to the other room.

(Structured + robust goal handling)
- Callbacks only store data (no goal publish inside callbacks)
- /goal_pose uses TRANSIENT_LOCAL + RELIABLE + depth=1 (latched)
- control_loop manages: init -> navigating -> finished
- goal republish burst + keepalive to avoid startup race
"""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


# 두 빈 방의 좌표 (x, y, yaw)
VACANT_ROOMS = [
    (7.53, 13.44, 0.77),    # Room A
    (-7.24, 13.26, 2.07),   # Room B
]

# 목표 지점 반경 내에서만 stop sign 인정
STOP_CHECK_DISTANCE_THRESHOLD = 5.0

# Robustness parameters
GOAL_BURST_DURATION = 1.0        # seconds: republish fast right after setting a goal
GOAL_KEEPALIVE_PERIOD = 1.0      # seconds: republish while navigating (slow)
ARRIVAL_DIST = 1.0               # meters: consider "arrived" if within this distance
ARRIVAL_TIMEOUT = 60.0           # seconds: if not arrived, re-burst goal


def yaw_to_quaternion_z_w(yaw: float) -> Tuple[float, float]:
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class GoToVacantRoom(Node):
    def __init__(self) -> None:
        super().__init__("go_to_vacant_room")

        # Inputs (from callbacks)
        self.latest_pose: Optional[PoseStamped] = None
        self.stop_sign_detected: bool = False

        # Mission state
        self.state = "WAIT_FOR_POSE"  # WAIT_FOR_POSE -> NAVIGATING -> FINISHED
        self.current_target_index: Optional[int] = None
        self.switched_room = False

        # Active goal tracking
        self.active_goal: Optional[PoseStamped] = None
        self.active_goal_xy: Optional[Tuple[float, float]] = None
        self.goal_sent_ns: Optional[int] = None
        self.last_goal_pub_ns: Optional[int] = None

        # Subscribers (store only)
        self.create_subscription(PoseStamped, "/go1_pose", self.pose_callback, 10)
        self.create_subscription(Bool, "/perception/stop_sign", self.stop_sign_callback, 10)

        # Publisher (latched)
        qos_goal = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)

        # Control loop
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("GoToVacantRoom started (structured). Waiting for /go1_pose...")

    # -----------------------------
    # Callbacks (store only)
    # -----------------------------
    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def stop_sign_callback(self, msg: Bool) -> None:
        # We only latch True; we can clear it after handling in control_loop
        if msg.data:
            self.stop_sign_detected = True

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
    def set_active_goal_by_index(self, idx: int) -> None:
        tx, ty, tyaw = VACANT_ROOMS[idx]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(tx)
        goal.pose.position.y = float(ty)
        goal.pose.position.z = 0.0

        z, w = yaw_to_quaternion_z_w(float(tyaw))
        goal.pose.orientation.z = z
        goal.pose.orientation.w = w

        self.current_target_index = idx
        self.active_goal = goal
        self.active_goal_xy = (float(tx), float(ty))

        now_ns = self._now_ns()
        self.goal_sent_ns = now_ns
        self.last_goal_pub_ns = None

        self.get_logger().info(f"Active goal set: Room index {idx} -> ({tx:.2f}, {ty:.2f}, yaw={tyaw:.2f})")

    def maybe_publish_active_goal(self) -> None:
        if self.active_goal is None:
            return

        now_ns = self._now_ns()
        sent_ns = self.goal_sent_ns or now_ns
        elapsed_s = (now_ns - sent_ns) / 1e9

        # Burst phase
        if elapsed_s <= GOAL_BURST_DURATION:
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            return

        # Keepalive phase
        if self.last_goal_pub_ns is None or (now_ns - self.last_goal_pub_ns) >= self._sec_to_ns(GOAL_KEEPALIVE_PERIOD):
            self.active_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(self.active_goal)
            self.last_goal_pub_ns = now_ns

    def active_goal_timeout(self) -> bool:
        if self.goal_sent_ns is None:
            return False
        return (self._now_ns() - self.goal_sent_ns) >= self._sec_to_ns(ARRIVAL_TIMEOUT)

    def dist_to_target(self) -> Optional[float]:
        if self.latest_pose is None or self.active_goal_xy is None:
            return None
        cx = self.latest_pose.pose.position.x
        cy = self.latest_pose.pose.position.y
        tx, ty = self.active_goal_xy
        return math.hypot(tx - cx, ty - cy)

    def choose_nearest_room_index(self) -> int:
        assert self.latest_pose is not None
        cx = self.latest_pose.pose.position.x
        cy = self.latest_pose.pose.position.y
        nearest_idx, _ = min(
            enumerate(VACANT_ROOMS),
            key=lambda item: math.hypot(item[1][0] - cx, item[1][1] - cy),
        )
        return nearest_idx

    def should_switch_rooms_now(self) -> bool:
        # Must have a stop-sign event latched
        if not self.stop_sign_detected:
            return False
        # Must have a target
        if self.current_target_index is None or self.active_goal_xy is None or self.latest_pose is None:
            return False
        # Only allow single switch
        if self.switched_room:
            return False

        cx = self.latest_pose.pose.position.x
        cy = self.latest_pose.pose.position.y
        tx, ty = self.active_goal_xy
        dist = math.hypot(tx - cx, ty - cy)

        # Only accept stop sign if close to the current target room
        if dist < STOP_CHECK_DISTANCE_THRESHOLD:
            self.get_logger().warn(
                f"STOP SIGN detected within range ({dist:.2f}m < {STOP_CHECK_DISTANCE_THRESHOLD}m). Switching rooms."
            )
            return True

        self.get_logger().info(
            f"Stop sign detected but ignored (too far: {dist:.2f}m > {STOP_CHECK_DISTANCE_THRESHOLD}m)."
        )
        return False

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self) -> None:
        if self.state == "FINISHED":
            return

        # Need pose to do anything
        if self.latest_pose is None:
            return

        # Initialize goal once
        if self.state == "WAIT_FOR_POSE":
            nearest_idx = self.choose_nearest_room_index()
            self.set_active_goal_by_index(nearest_idx)
            self.state = "NAVIGATING"
            return

        # NAVIGATING
        if self.state == "NAVIGATING":
            # Always keep goal alive (prevents startup/subscriber races)
            self.maybe_publish_active_goal()

            # Handle stop sign event (latched)
            if self.should_switch_rooms_now():
                new_idx = 1 - int(self.current_target_index)
                self.switched_room = True
                self.set_active_goal_by_index(new_idx)
                # consume the event
                self.stop_sign_detected = False
                return

            # If we didn't switch, still clear the latched stop event so it doesn't keep retriggering logs
            if self.stop_sign_detected:
                self.stop_sign_detected = False

            # Arrival check
            dist = self.dist_to_target()
            if dist is not None and dist <= ARRIVAL_DIST:
                self.get_logger().info(f"Arrived at Room index {self.current_target_index} (dist={dist:.2f}m).")
                self.state = "FINISHED"
                return

            # Timeout recovery: re-burst
            if self.active_goal_timeout():
                self.get_logger().warn("Arrival timeout. Re-bursting current goal.")
                self.goal_sent_ns = self._now_ns()
                self.last_goal_pub_ns = None
            return


def main(args=None):
    rclpy.init(args=args)
    node = GoToVacantRoom()
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
