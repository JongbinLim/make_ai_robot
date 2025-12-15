#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Float32
from sensor_msgs.msg import CameraInfo

# -----------------------------
# Mission Constants
# -----------------------------
ROOM_LIST = [
    ("Room 1", -5.14, 15.62, 1.29),
    ("Room 2", -6.78, 12.53, 2.55),
    ("Room 3", -9.27, 11.15, -2.64),
    ("Room 4", -9.40, 4.84, 2.43),
    ("Room 5", -2.86, -0.29, -0.81),
    ("Room 6", 2.81, -0.33, -2.26),
    ("Room 7", 9.38, 4.90, 0.23),
    ("Room 8", 9.49, 11.11, -0.37),
    ("Room 9", 5.08, 15.58, 1.36),
    ("Room 10", 5.17, 15.33, 1.88),
    ("Room 11", 2.82, -10.11, -2.27),
    ("Room 12", -2.83, -10.34, -0.69),
    ("Room 13", -2.72, -16.58, -1.00),
    ("Room 14", 2.83, -16.51, -1.93),
    ("Room 15", 7.23, -22.90, 1.17),
    ("Room 16", 5.45, -26.54, 1.56),
    ("Room 17", -6.90, -24.97, -2.08),
    ("Room 18", -7.02, -22.93, 1.88),
    ("Room 19", -7.08, -8.96, 2.17),
    ("Room 20", 7.33, -8.83, 0.88),
]

EDIBLE_SET = {"apple", "banana", "pizza"}

# Tuning Parameters
CONTROL_PERIOD = 0.2          # 5Hz (빠른 반응)
GOAL_RESEND_PERIOD = 3.0      # 목표 재전송 주기 (Smart Publish)
BARK_DISTANCE_THRESHOLD = 0.5 # 짖는 거리
ARRIVAL_THRESHOLD = 0.5       # 방 도착 판정

# Centering & Approach
CENTER_TOL_PX = 40.0          # 화면 중앙 허용 오차 (pixel)
MAX_YAW_STEP = math.radians(20)
CENTERING_STABLE_TICKS = 3    # 몇 번 연속 중앙에 있어야 안정된 것으로 볼지
IMG_WIDTH_FALLBACK = 640.0    # CameraInfo 없을 때 기본값

class FoodSmartSearcher(Node):
    def __init__(self):
        super().__init__('food_smart_searcher')
        
        # State Machine: PATROL -> CENTERING -> APPROACH -> WAITING_BARK -> COMPLETE
        self.state = "PATROL"
        self._state_entered = True

        # -----------------------------
        # Data Containers
        # -----------------------------
        self.current_pose: PoseStamped | None = None
        self.camera_info: CameraInfo | None = None
        
        self.detected_labels = set()
        self.food_u = -1.0      # 음식의 화면상 가로 좌표 (pixel)
        self.food_depth = -1.0  # 음식까지의 거리 (m)
        
        self.last_detection_time = None
        self.center_stable_cnt = 0
        
        # Frozen Target (맵 상의 음식 절대 좌표)
        self.target_map_x = None
        self.target_map_y = None

        # Smart Goal Publishing Variables
        self.last_published_target = None  # (x, y, yaw)
        self.last_goal_pub_time = None

        # Room Index
        self.room_index = 0

        # -----------------------------
        # Subscribers & Publishers
        # -----------------------------
        qos = QoSProfile(depth=10)
        
        self.pose_sub = self.create_subscription(PoseStamped, "/go1_pose", self.pose_callback, qos)
        self.labels_sub = self.create_subscription(String, "/detections/labels", self.labels_callback, qos)
        self.dist_sub = self.create_subscription(Float32, "/detections/distance", self.distance_callback, qos)
        self.speech_sub = self.create_subscription(String, "/robot_dog/speech", self.speech_callback, qos)
        self.caminfo_sub = self.create_subscription(CameraInfo, "/camera_top/camera_info", self.caminfo_callback, qos)

        # [중요] 음식의 중심점(u, v) 혹은 (u, depth)를 주는 토픽이 필요함.
        # 기존 코드에는 없었으나 Visual Servoing을 위해 추가 권장.
        # 만약 없다면 self.food_u를 이미지 중앙값으로 강제해야 함.
        self.center_sub = self.create_subscription(PointStamped, "/detections/food_center", self.center_callback, qos)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info("Food Smart Searcher Started.")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def labels_callback(self, msg: String):
        labels = [s.strip().lower() for s in msg.data.split(",") if s.strip()]
        self.detected_labels = set(labels)
        if any(e in self.detected_labels for e in EDIBLE_SET):
            self.last_detection_time = self.get_clock().now()

    def distance_callback(self, msg: Float32):
        # 기존 distance 토픽 사용
        self.food_depth = msg.data

    def center_callback(self, msg: PointStamped):
        # NurseOrbiter처럼 중심점(u) 정보를 받음
        self.food_u = msg.point.x

    def caminfo_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def speech_callback(self, msg: String):
        if self.state == "WAITING_BARK" and "bark" in msg.data.lower():
            self.get_logger().info("!!! MISSION COMPLETE (Bark Detected) !!!")
            self.set_state("COMPLETE")

    # -----------------------------
    # Smart Goal Publishing (핵심 기능)
    # -----------------------------
    def publish_goal_smart(self, x, y, yaw):
        now = self.get_clock().now()
        
        # 1. 변경 감지
        is_different = False
        if self.last_published_target is None:
            is_different = True
        else:
            lx, ly, lyaw = self.last_published_target
            diff = math.hypot(lx - x, ly - y) + abs(lyaw - yaw)
            if diff > 0.05: # 5cm 또는 0.05rad 이상 차이날 때만 갱신
                is_different = True

        # 2. 타임아웃 (Keep-alive)
        is_timeout = False
        if self.last_goal_pub_time is not None:
            elapsed = (now - self.last_goal_pub_time).nanoseconds * 1e-9
            if elapsed > GOAL_RESEND_PERIOD:
                is_timeout = True
        else:
            is_timeout = True

        if is_different or is_timeout:
            self._publish_goal_force(x, y, yaw)
            self.last_published_target = (x, y, yaw)
            self.last_goal_pub_time = now

    def _publish_goal_force(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        goal.pose.orientation.w = cy
        goal.pose.orientation.z = sy
        self.goal_pub.publish(goal)

    # -----------------------------
    # Helpers
    # -----------------------------
    def set_state(self, s):
        if self.state != s:
            self.get_logger().info(f"State Transition: {self.state} -> {s}")
            self.state = s
            self._state_entered = True
            self.last_published_target = None # 상태 변경 시 즉시 목표 전송

    def get_yaw(self):
        if not self.current_pose: return 0.0
        q = self.current_pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def is_detected_fresh(self):
        if self.last_detection_time is None: return False
        elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds * 1e-9
        return elapsed < 2.0 # 2초 이내 감지됨

    def get_image_center_u(self):
        if self.camera_info and self.camera_info.width > 0:
            return self.camera_info.width * 0.5
        return IMG_WIDTH_FALLBACK * 0.5

    def pixel_to_bearing(self, u_px):
        # Pinhole Camera Model (NurseOrbiter 방식)
        if self.camera_info and len(self.camera_info.k) >= 9:
            fx = self.camera_info.k[0]
            cx = self.camera_info.k[2]
            if fx > 1e-6:
                return math.atan2((cx - u_px), fx)
        
        # Fallback (FOV 80도 가정)
        w = IMG_WIDTH_FALLBACK
        hfov = math.radians(80.0)
        ratio = (u_px - (w * 0.5)) / (w * 0.5)
        return -ratio * (hfov * 0.5)

    def freeze_food_position(self, curr_x, curr_y, curr_yaw):
        if self.food_depth <= 0.1: return False
        
        # 화면상 위치를 각도로 변환
        bearing = self.pixel_to_bearing(self.food_u)
        abs_yaw = curr_yaw + bearing
        
        # 맵 상의 절대 좌표 계산 (Map Freeze)
        self.target_map_x = curr_x + self.food_depth * math.cos(abs_yaw)
        self.target_map_y = curr_y + self.food_depth * math.sin(abs_yaw)
        self.get_logger().info(f"Food Frozen at Map: ({self.target_map_x:.2f}, {self.target_map_y:.2f})")
        return True

    # -----------------------------
    # Main Loop
    # -----------------------------
    def control_loop(self):
        if self.current_pose is None: return

        curr_x = self.current_pose.pose.position.x
        curr_y = self.current_pose.pose.position.y
        curr_yaw = self.get_yaw()

        # [STATE 1] PATROL
        if self.state == "PATROL":
            # 음식 감지 확인
            if self.is_detected_fresh():
                self.get_logger().info("Food Detected! Switching to CENTERING.")
                self.set_state("CENTERING")
                self.center_stable_cnt = 0
                return
            
            # 모든 방 순찰 완료
            if self.room_index >= len(ROOM_LIST):
                self.get_logger().info("Patrol Finished. No food found.")
                self.set_state("COMPLETE")
                return

            # 방으로 이동
            target = ROOM_LIST[self.room_index]
            dist = math.hypot(target[1] - curr_x, target[2] - curr_y)
            
            self.publish_goal_smart(target[1], target[2], target[3])
            
            if dist < ARRIVAL_THRESHOLD:
                self.get_logger().info(f"Arrived at {target[0]}")
                self.room_index += 1

        # [STATE 2] CENTERING (Visual Servoing)
        elif self.state == "CENTERING":
            if not self.is_detected_fresh():
                self.get_logger().warn("Lost food during centering -> Back to PATROL")
                self.set_state("PATROL")
                return

            # 화면 중앙값과 현재 음식 위치 차이 계산
            center_u = self.get_image_center_u()
            # 만약 food_u가 업데이트 안 되었다면 중앙이라고 가정(fallback)
            u = self.food_u if self.food_u > 0 else center_u 
            
            diff_u = u - center_u
            
            # 목표 각도 계산 (NurseOrbiter 방식)
            bearing = self.pixel_to_bearing(u)
            target_yaw = curr_yaw + max(min(bearing, MAX_YAW_STEP), -MAX_YAW_STEP) # Clamp
            
            # 제자리 회전 명령 (Smart Publish)
            self.publish_goal_smart(curr_x, curr_y, target_yaw)

            # 중앙 정렬 확인
            if abs(diff_u) < CENTER_TOL_PX:
                self.center_stable_cnt += 1
            else:
                self.center_stable_cnt = 0

            # 안정적으로 중앙에 왔다면 좌표 고정(Freeze) 후 접근
            if self.center_stable_cnt >= CENTERING_STABLE_TICKS:
                if self.freeze_food_position(curr_x, curr_y, curr_yaw):
                    self.set_state("APPROACH")

        # [STATE 3] APPROACH (Frozen Coordinates)
        elif self.state == "APPROACH":
            # 거리가 가까우면 멈춤
            dist_to_food = math.hypot(self.target_map_x - curr_x, self.target_map_y - curr_y)
            
            if dist_to_food <= BARK_DISTANCE_THRESHOLD:
                self.get_logger().info("Close enough. Waiting for Bark...")
                # 제자리에 멈춰서 대기
                self.publish_goal_smart(curr_x, curr_y, curr_yaw)
                self.set_state("WAITING_BARK")
                return

            # 저장된 좌표로 계속 이동 (인식이 끊겨도 이동함)
            # 바라보는 방향은 목표 지점을 향하도록
            goal_yaw = math.atan2(self.target_map_y - curr_y, self.target_map_x - curr_x)
            self.publish_goal_smart(self.target_map_x, self.target_map_y, goal_yaw)

        # [STATE 4] WAITING_BARK
        elif self.state == "WAITING_BARK":
            # Speech Callback에서 "COMPLETE"로 변경해줌
            pass

        # [STATE 5] COMPLETE
        elif self.state == "COMPLETE":
            pass

def main(args=None):
    rclpy.init(args=args)
    node = FoodSmartSearcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
