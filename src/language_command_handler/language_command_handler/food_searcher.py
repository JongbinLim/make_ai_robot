#!/usr/bin/env python3
"""
Node: food_searcher

Behavior:
1. Patrol predefined rooms.
2. Listen to perception outputs (labels + distance).
3. If edible object is visible but far:
   - Move forward incrementally toward it.
4. When perception barks (distance <= threshold):
   - Stop immediately and end mission.
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

# -----------------------------
# Mission parameters
# -----------------------------
ROOM_LIST = [
    ("Nurse Station", -6.25, -25.0, 3.13),
    ("Room 1", 5.53, -25.9, -1.57),
    ("Room 2", 6.51, -23.0, 0.0),
    ("Room 3", -6.51, -23.0, 3.13),
    ("Room 4", 2.84, -16.0, -1.57),
    ("Room 5", -2.84, -16.0, -1.57),
    ("Room 6", 2.84, -10.0, -1.57),
    ("Room 7", -2.84, -10.0, -1.57),
    ("Room 8", 6.51, -9.0, 0.0),
    ("Room 9", -6.51, -9.0, 3.13),
]

EDIBLE_SET = {"apple", "banana", "pizza"}

BARK_DISTANCE_THRESHOLD = 0.4   # must match perception
APPROACH_STEP = 0.8             # meters
ARRIVAL_THRESHOLD = 0.5         # room arrival


class FoodSearcher(Node):
    def __init__(self):
        super().__init__('food_searcher')

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.pose_sub = self.create_subscription(
            PoseStamped, "/go1_pose", self.pose_callback, 10
        )

        self.speech_sub = self.create_subscription(
            String, "/robot_dog/speech", self.speech_callback, 10
        )

        self.labels_sub = self.create_subscription(
            String, "/detections/labels", self.labels_callback, 10
        )

        self.dist_sub = self.create_subscription(
            Float32, "/detections/distance", self.distance_callback, 10
        )

        # -----------------------------
        # Publisher
        # -----------------------------
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # -----------------------------
        # Internal State
        # -----------------------------
        self.current_pose = None
        self.detected_labels = set()
        self.target_distance = -1.0

        self.room_index = 0
        self.is_navigating = False
        self.food_found = False
        self.approaching_food = False

        # Control loop (1 Hz)
        self.timer = self.create_timer(1.0, self.control_loop)

        self.get_logger().info("Food Searcher Node Initialized.")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def labels_callback(self, msg: String):
        labels = [s.strip().lower() for s in msg.data.split(",") if s.strip()]
        self.detected_labels = set(labels)

    def distance_callback(self, msg: Float32):
        self.target_distance = msg.data

    def speech_callback(self, msg: String):
        """
        Perception node barks ONLY when distance <= threshold.
        """
        if self.food_found:
            return

        if "bark" in msg.data.lower():
            self.get_logger().info("!!! FOOD FOUND (BARK DETECTED) !!!")
            self.food_found = True
            self.stop_robot()

    # -----------------------------
    # Helpers
    # -----------------------------
    def stop_robot(self):
        if self.current_pose:
            stop_goal = self.current_pose
            stop_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(stop_goal)
        self.is_navigating = False
        self.approaching_food = False

    def publish_room_goal(self, room_data):
        name, x, y, yaw = room_data

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0

        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(goal)
        self.get_logger().info(f"Navigating to: {name}")

    def publish_forward_step(self, step):
        if self.current_pose is None:
            return

        q = self.current_pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z),
            1.0 - 2.0 * (q.z * q.z)
        )

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        gx = cx + step * math.cos(yaw)
        gy = cy + step * math.sin(yaw)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0

        goal.pose.orientation = self.current_pose.pose.orientation
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Approaching food: step {step:.2f} m (dist={self.target_distance:.2f})"
        )

    # -----------------------------
    # Main control loop
    # -----------------------------
    def control_loop(self):
        if self.food_found or self.current_pose is None:
            return

        edible_visible = any(e in self.detected_labels for e in EDIBLE_SET)
        distance_valid = self.target_distance > 0

        # -----------------------------
        # APPROACH MODE (override patrol)
        # -----------------------------
        if edible_visible and distance_valid and self.target_distance > BARK_DISTANCE_THRESHOLD:
            self.approaching_food = True
            self.is_navigating = False
            self.publish_forward_step(APPROACH_STEP)
            return

        if self.approaching_food and distance_valid and self.target_distance <= BARK_DISTANCE_THRESHOLD:
            self.get_logger().info("Close to food. Waiting for bark...")
            return

        # -----------------------------
        # ROOM PATROL MODE
        # -----------------------------
        if self.room_index >= len(ROOM_LIST):
            self.get_logger().info("All rooms searched. Mission complete.")
            self.food_found = True
            return

        target_room = ROOM_LIST[self.room_index]
        tx, ty = target_room[1], target_room[2]
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        dist = math.hypot(tx - cx, ty - cy)

        if not self.is_navigating:
            self.publish_room_goal(target_room)
            self.is_navigating = True

        else:
            if dist < ARRIVAL_THRESHOLD:
                self.get_logger().info(f"Arrived at {target_room[0]}")
                self.room_index += 1
                self.is_navigating = False


def main(args=None):
    rclpy.init(args=args)
    node = FoodSearcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Food Searcher stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
