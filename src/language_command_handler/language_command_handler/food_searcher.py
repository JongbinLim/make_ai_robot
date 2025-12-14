#!/usr/bin/env python3
"""
Node: food_searcher
Description:
1. Navigates through a predefined list of rooms.
2. Listens to '/robot_dog/speech'.
3. If 'bark' is detected (meaning Perception Node found edible food), the robot stops immediately.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# 미션에 정의된 방 좌표 목록 (x, y, yaw)
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
    ("Room 10", 2.84, 0.0, -1.57),
    ("Room 11", -2.84, 0.0, -1.57),
    ("Room 12", 8.88, 4.77, 0.0),
    ("Room 13", -8.88, 4.77, 0.0),
    ("Room 14", 8.88, 11.2, 0.0),
    ("Room 15", -8.88, 11.2, 3.13),
    ("Room 16", 6.68, 11.9, 1.57),
    ("Room 17", -6.68, 11.9, 1.57),
    ("Room 18", 5.22, 14.9, 1.57),
    ("Room 19", -5.22, 14.9, 1.57),
]

class FoodSearcher(Node):
    def __init__(self):
        super().__init__('food_searcher')

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/go1_pose", self.pose_callback, 10
        )
        # Perception 노드가 음식을 발견하고 짖는 것을 감지
        self.speech_sub = self.create_subscription(
            String, "/robot_dog/speech", self.speech_callback, 10
        )

        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Internal State
        self.current_pose = None
        self.room_index = 0
        self.is_navigating = False
        self.food_found = False
        
        # 목표지점 도착 판단 거리 (미터)
        self.arrival_threshold = 0.5 

        # 주기적으로 상태 체크 및 명령 발행 (1Hz)
        self.timer = self.create_timer(1.0, self.control_loop)

        self.get_logger().info("Food Searcher Node Initialized. Ready to hunt for food!")

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def speech_callback(self, msg: String):
        """
        Perception 노드가 'bark'를 보내면 음식을 찾은 것으로 간주하고 정지
        """
        if self.food_found:
            return

        if "bark" in msg.data.lower():
            self.get_logger().info("!!! FOOD FOUND (BARK DETECTED) !!! Stopping robot.")
            self.food_found = True
            self.stop_robot()

    def stop_robot(self):
        """현재 위치를 목표로 재전송하여 로봇 정지"""
        if self.current_pose:
            stop_goal = self.current_pose
            stop_goal.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(stop_goal)
        self.is_navigating = False

    def publish_goal(self, room_data):
        """특정 방으로 이동 명령 발행"""
        name, x, y, yaw = room_data
        
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0

        # Yaw to Quaternion conversion
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(goal)
        self.get_logger().info(f"Navigating to: {name}")

    def control_loop(self):
        # 1. 이미 음식을 찾았거나, 위치 정보가 없으면 패스
        if self.food_found or self.current_pose is None:
            return

        # 2. 모든 방을 다 돌았으면 종료
        if self.room_index >= len(ROOM_LIST):
            self.get_logger().info("Searched all rooms but found nothing...")
            self.food_found = True # 루프 종료를 위해 true 처리
            return

        # 3. 현재 목표 방 정보 가져오기
        target_room = ROOM_LIST[self.room_index]
        tx, ty = target_room[1], target_room[2]
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        dist = math.hypot(tx - cx, ty - cy)

        # 4. 내비게이션 로직
        if not self.is_navigating:
            # 이동 시작
            self.publish_goal(target_room)
            self.is_navigating = True
        
        else:
            # 이동 중: 도착했는지 확인
            if dist < self.arrival_threshold:
                self.get_logger().info(f"Arrived at {target_room[0]}. Checking for food...")
                # 도착 후 잠시 대기하거나 즉시 다음 방으로 넘어갈 수 있음.
                # 여기서는 바로 다음 방 인덱스로 넘기지만, 
                # Perception은 이동 중에도 계속 돌고 있으므로 발견 시 speech_callback이 잡음.
                self.room_index += 1
                self.is_navigating = False # 다음 루프에서 다음 방으로 이동 시작

def main(args=None):
    rclpy.init(args=args)
    node = FoodSearcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Food Searcher stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
