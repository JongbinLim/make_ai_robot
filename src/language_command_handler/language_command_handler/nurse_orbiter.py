#!/usr/bin/env python3
"""
Node: nurse_orbiter
Description:
1. 지정된 Break Room 좌표로 이동합니다.
2. Perception 노드를 통해 간호사(person)를 찾습니다.
3. 간호사를 중심으로 원형으로 회전(Orbit)합니다.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

# 미션 상수 설정
TARGET_ROOM_POSE = (-6.25, -25.0, 3.13) # (x, y, yaw)
TARGET_LABEL = "person"                 # YOLO에서 감지되는 라벨명 (간호사)
ORBIT_RADIUS_OFFSET = 0.0               # 인식된 거리보다 더 멀리 돌고 싶으면 값을 추가 (미터)
ORBIT_POINTS = 12                       # 원을 몇 개의 점으로 나눌지 (많을수록 부드러움)
WAYPOINT_TOLERANCE = 0.5                # 각 웨이포인트 도착 판정 거리

class NurseOrbiter(Node):
    def __init__(self):
        super().__init__('nurse_orbiter')

        # 상태 정의
        self.state = "GO_TO_ROOM" # GO_TO_ROOM -> SEARCHING -> ORBITING -> FINISHED
        self.current_pose = None
        self.detected_labels = []
        self.target_distance = -1.0
        
        # Orbit 관련 변수
        self.nurse_pose = None # (x, y)
        self.orbit_waypoints = []
        self.current_waypoint_idx = 0
        self.orbit_radius = 1.5 # 기본값 (탐색 실패 시 안전거리)

        # QoS 설정
        qos_profile = QoSProfile(depth=10)

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            "/go1_pose", 
            self.pose_callback, 
            qos_profile
        )
        
        self.label_sub = self.create_subscription(
            String, 
            "/detections/labels", 
            self.label_callback, 
            qos_profile
        )
        
        # Perception Node에서 거리를 받아옴 (topic 이름은 perception_node 구현에 따라 다를 수 있으나 관례상 추정)
        # perception_node.py의 self.pub_distance에 해당하는 토픽
        self.dist_sub = self.create_subscription(
            Float32,
            "/detections/distance", 
            self.distance_callback,
            qos_profile
        )

        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Main Control Loop (0.5초 주기)
        self.timer = self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info("Nurse Orbiter Node Started. Destination: Break Room")

    def pose_callback(self, msg):
        self.current_pose = msg

    def label_callback(self, msg):
        # "person, chair, table" 형태의 문자열을 리스트로 파싱
        self.detected_labels = [label.strip() for label in msg.data.split(',')]

    def distance_callback(self, msg):
        self.target_distance = msg.data

    def get_yaw_from_pose(self, pose):
        # 쿼터니언을 Yaw(라디안)로 변환
        q = pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # Yaw to Quaternion (간략화)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        goal.pose.orientation.w = cy
        goal.pose.orientation.z = sy
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0

        self.goal_pub.publish(goal)

    def generate_orbit_path(self, center_x, center_y, radius):
        """간호사 위치(Center)를 중심으로 원형 웨이포인트 생성"""
        waypoints = []
        for i in range(ORBIT_POINTS):
            angle = (2 * math.pi / ORBIT_POINTS) * i
            # 현재 로봇 위치에서 시작하도록 위상을 조정할 수도 있으나, 여기선 0도부터 시작
            wx = center_x + radius * math.cos(angle)
            wy = center_y + radius * math.sin(angle)
            w_yaw = angle + (math.pi / 2)
            
            waypoints.append((wx, wy, w_yaw))
        return waypoints

    def control_loop(self):
        if self.current_pose is None:
            return

        curr_x = self.current_pose.pose.position.x
        curr_y = self.current_pose.pose.position.y
        curr_yaw = self.get_yaw_from_pose(self.current_pose)

        # ---------------------------------------------------------
        # 1. 방으로 이동 (Move to Room)
        # ---------------------------------------------------------
        if self.state == "GO_TO_ROOM":
            tx, ty, tyaw = TARGET_ROOM_POSE
            dist = math.hypot(tx - curr_x, ty - curr_y)

            if dist > 1.0: # 도착 임계값 (방 입구 근처면 충분)
                self.publish_goal(tx, ty, tyaw)
                self.get_logger().info_once("Moving to the Break Room...")
            else:
                self.get_logger().info("Arrived at room. Start searching for nurse...")
                self.state = "SEARCHING"

        # ---------------------------------------------------------
        # 2. 간호사 탐색 (Search)
        # ---------------------------------------------------------
        elif self.state == "SEARCHING":
            # 간호사(person)가 감지되고, 유효한 거리정보가 들어오는지 확인
            if TARGET_LABEL in self.detected_labels and self.target_distance > 0:
                self.get_logger().info(f"Nurse detected! Distance: {self.target_distance:.2f}m")
                
                # 간호사의 절대 위치 추정 (현재 로봇 위치 + 거리 * 방향)
                # Perception Node가 카메라 정면 기준 거리를 준다고 가정
                nurse_x = curr_x + self.target_distance * math.cos(curr_yaw)
                nurse_y = curr_y + self.target_distance * math.sin(curr_yaw)
                self.nurse_pose = (nurse_x, nurse_y)
                
                # 회전 반경 설정 (발견된 거리 그대로 유지하거나 약간 조정)
                self.orbit_radius = self.target_distance + ORBIT_RADIUS_OFFSET
                
                # 웨이포인트 생성
                self.orbit_waypoints = self.generate_orbit_path(nurse_x, nurse_y, self.orbit_radius)
                self.state = "ORBITING"
                self.get_logger().info(f"Orbit path generated with {len(self.orbit_waypoints)} points.")
            
            else:
                # 못 찾았으면 제자리 회전하며 탐색 (또는 방 좌표로 계속 이동)
                self.get_logger().info_once("Searching for nurse...")
                # 제자리 회전 명령 (현재 위치에서 yaw만 변경) - 여기선 단순화를 위해 기존 goal 유지
                # 필요 시 제자리 회전 로직 추가 가능

        # ---------------------------------------------------------
        # 3. 회전 (Orbit)
        # ---------------------------------------------------------
        elif self.state == "ORBITING":
            if self.current_waypoint_idx >= len(self.orbit_waypoints):
                self.get_logger().info("Orbit Mission Completed!")
                self.state = "FINISHED"
                return

            # 현재 목표 웨이포인트
            wx, wy, wyaw = self.orbit_waypoints[self.current_waypoint_idx]
            dist_to_wp = math.hypot(wx - curr_x, wy - curr_y)

            # 웨이포인트 도착 확인
            if dist_to_wp < WAYPOINT_TOLERANCE:
                self.current_waypoint_idx += 1
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx}/{ORBIT_POINTS}")
            else:
                # 목표 지점으로 이동 명령
                self.publish_goal(wx, wy, wyaw)

        elif self.state == "FINISHED":
            pass

def main(args=None):
    rclpy.init(args=args)
    node = NurseOrbiter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
