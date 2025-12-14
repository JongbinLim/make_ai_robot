#!/usr/bin/env python3
"""
Navigate to the nearest room. 
If a stop sign is detected strictly near the target room, switch to the other room.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

# 두 빈 방의 좌표 (x, y, yaw)
VACANT_ROOMS = [
    (6.68, 11.9, 1.57),   # Room A
    (-6.68, 11.9, 1.57),  # Room B
]

# Stop 표지판을 유효하게 인식할 거리 임계값 (미터)
# 목표 지점 반경 3.0m 안으로 들어왔을 때만 표지판을 검사함
STOP_CHECK_DISTANCE_THRESHOLD = 3.0 

class GoToVacantRoom(Node):
    def __init__(self) -> None:
        super().__init__("go_to_vacant_room")

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/go1_pose", self.pose_callback, 10
        )
        # Perception 노드에서 Stop sign 감지 여부를 받음
        self.stop_sub = self.create_subscription(
            Bool, "/perception/stop_sign", self.stop_sign_callback, 10
        )

        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.latest_pose = None
        self.current_target_index = None # 현재 목표로 삼은 방의 인덱스
        self.goal_sent = False
        self.switched_room = False  # 방을 바꿨는지 여부 (한 번만 바꾸기 위해)

        self.get_logger().info("GoToVacantRoom initialized with Distance Check.")

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        
        # 아직 목표를 설정하지 않았다면, 가장 가까운 방으로 초기 설정
        if not self.goal_sent and not self.switched_room:
            self.set_initial_nearest_goal()

    def stop_sign_callback(self, msg: Bool) -> None:
        """Perception 노드가 Stop 표지판을 보고 True를 보내면 실행"""
        # 1. Stop 사인이 아니거나, 아직 위치/목표 정보가 없으면 무시
        if msg.data is False:
            return
        if self.latest_pose is None or self.current_target_index is None:
            return
        
        # 이미 방을 바꾼 상태라면 더 이상 체크하지 않음
        if self.switched_room:
            return

        # 현재 로봇 위치와 '현재 목표 방' 사이의 거리 계산
        cx = self.latest_pose.pose.position.x
        cy = self.latest_pose.pose.position.y
        
        target_x, target_y, _ = VACANT_ROOMS[self.current_target_index]
        
        dist_to_target = math.hypot(target_x - cx, target_y - cy)

        # 2. 거리가 임계값(예: 3.0m) 이내일 때만 Stop 사인 인정
        if dist_to_target < STOP_CHECK_DISTANCE_THRESHOLD:
            self.get_logger().warn(
                f"STOP SIGN DETECTED within range ({dist_to_target:.2f}m)! Switching rooms..."
            )
            self.switch_to_other_room()
        else:
            # 거리가 멀다면, 가는 길에 다른 방의 표지판을 본 것으로 간주하고 무시
            self.get_logger().info(
                f"Stop sign detected but ignored. Too far from target ({dist_to_target:.2f}m > {STOP_CHECK_DISTANCE_THRESHOLD}m)."
            )

    def set_initial_nearest_goal(self) -> None:
        if self.latest_pose is None:
            return

        cx = self.latest_pose.pose.position.x
        cy = self.latest_pose.pose.position.y

        # 가장 가까운 방 찾기 (인덱스 저장)
        nearest_idx, nearest_room = min(
            enumerate(VACANT_ROOMS), 
            key=lambda item: math.hypot(item[1][0] - cx, item[1][1] - cy)
        )

        self.current_target_index = nearest_idx
        self.publish_goal(nearest_room)
        self.goal_sent = True
        self.get_logger().info(f"Initial Goal: Room index {nearest_idx}")

    def switch_to_other_room(self) -> None:
        # 방이 두 개뿐이므로, 현재 인덱스가 0이면 1로, 1이면 0으로 변경
        new_idx = 1 - self.current_target_index
        new_room = VACANT_ROOMS[new_idx]

        self.current_target_index = new_idx
        self.switched_room = True  # 중복 스위칭 방지
        
        self.publish_goal(new_room)
        self.get_logger().info(f"Goal Switched: Now going to Room index {new_idx}")

    def publish_goal(self, room_data) -> None:
        tx, ty, tyaw = room_data

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = tx
        goal.pose.position.y = ty
        goal.pose.position.z = 0.0

        goal.pose.orientation.z = math.sin(tyaw / 2.0)
        goal.pose.orientation.w = math.cos(tyaw / 2.0)

        self.goal_pub.publish(goal)

def main(args=None):
    rclpy.init(args=args)
    node = GoToVacantRoom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
