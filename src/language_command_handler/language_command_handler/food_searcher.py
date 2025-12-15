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
    ("Room 7", 9.38, 4.90, 0.23),
    ("Room 1", -5.14, 15.62, 1.29), 
    ("Room 2", -6.78, 12.53, 2.55),
    ("Room 3", -9.27, 11.15, -2.64),
    ("Room 4", -9.40, 4.84, 2.43),
    ("Room 5", -2.86, -0.29, -0.81),
    ("Room 6", 2.81, -0.33, -2.26),
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
CONTROL_PERIOD = 0.2          
GOAL_RESEND_PERIOD = 2.0      
BARK_DISTANCE_THRESHOLD = 0.6 
ARRIVAL_THRESHOLD = 0.6       

# Centering & Approach
CENTER_TOL_PX = 60.0          
MAX_YAW_STEP = math.radians(25) 
CENTERING_STABLE_TICKS = 2     
IMG_WIDTH_FALLBACK = 640.0
DETECTION_TIMEOUT = 4.0        
CENTERING_TIMEOUT_SEC = 10.0  

# [NEW] 짖는 주기 (초 단위)
BARK_INTERVAL = 0.1  

class FoodSmartSearcher(Node):
    def __init__(self):
        super().__init__('food_smart_searcher')
        
        # State Machine
        self.state = "PATROL"
        self.state_start_time = self.get_clock().now()

        # Data Containers
        self.current_pose: PoseStamped | None = None
        self.camera_info: CameraInfo | None = None
        
        self.detected_labels = set()
        self.food_u = -1.0      
        self.food_depth = -1.0 
        
        self.last_detection_time = None
        self.center_stable_cnt = 0
        
        # Frozen Target
        self.target_map_x = None
        self.target_map_y = None

        # Smart Goal Publishing Variables
        self.last_published_target = None 
        self.last_goal_pub_time = None

        self.room_index = 0
        self.patrol_finish_timer = None

        # [NEW] Barking Timer Variable
        self.last_bark_time = None

        # Subscribers & Publishers
        qos = QoSProfile(depth=10)
        
        self.pose_sub = self.create_subscription(PoseStamped, "/go1_pose", self.pose_callback, qos)
        self.labels_sub = self.create_subscription(String, "/detections/labels", self.labels_callback, qos)
        self.dist_sub = self.create_subscription(Float32, "/detections/distance", self.distance_callback, qos)
        self.caminfo_sub = self.create_subscription(CameraInfo, "/camera_top/camera_info", self.caminfo_callback, qos)
        self.center_sub = self.create_subscription(PointStamped, "/detections/food_center", self.center_callback, qos)
        
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.voice_pub = self.create_publisher(String, "/robot_dog/speech", 10)
        
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info("Food Smart Searcher Started: Continuous Bark Mode.")

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
        if msg.data > 0.1: 
            self.food_depth = msg.data

    def center_callback(self, msg: PointStamped):
        self.food_u = msg.point.x

    def caminfo_callback(self, msg: CameraInfo):
        self.camera_info = msg

    # -----------------------------
    # Smart Goal Publishing
    # -----------------------------
    def publish_goal_smart(self, x, y, yaw):
        now = self.get_clock().now()
        
        is_different = False
        
        if self.last_published_target is None:
            is_different = True
        else:
            lx, ly, lyaw = self.last_published_target
            
            dist_diff = math.hypot(lx - x, ly - y)
            yaw_diff = abs(lyaw - yaw)

            if dist_diff > 0.03 or yaw_diff > 0.03: 
                is_different = True

        is_timeout = False
        if self.last_goal_pub_time is not None:
            elapsed = (now - self.last_goal_pub_time).nanoseconds * 1e-9
            if elapsed > 0.5: 
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
            self.state_start_time = self.get_clock().now() 
            self.last_published_target = None
            
            if s == "PATROL":
                self.food_u = -1.0
                self.food_depth = -1.0
                self.patrol_finish_timer = None 
                self.last_bark_time = None # 상태 초기화 시 짖는 타이머도 리셋

    def get_yaw(self):
        if not self.current_pose: return 0.0
        q = self.current_pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def is_detected_fresh(self):
        if self.last_detection_time is None: return False
        elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds * 1e-9
        return elapsed < DETECTION_TIMEOUT 

    def get_image_center_u(self):
        if self.camera_info and self.camera_info.width > 0:
            return self.camera_info.width * 0.5
        return IMG_WIDTH_FALLBACK * 0.5

    def pixel_to_bearing(self, u_px):
        if self.camera_info and len(self.camera_info.k) >= 9:
            fx = self.camera_info.k[0]
            cx = self.camera_info.k[2]
            if fx > 1e-6:
                return math.atan2((cx - u_px), fx)
        
        w = IMG_WIDTH_FALLBACK
        hfov = math.radians(80.0)
        ratio = (u_px - (w * 0.5)) / (w * 0.5)
        return -ratio * (hfov * 0.5)

    def freeze_food_position(self, curr_x, curr_y, curr_yaw):
        if self.food_depth <= 0.1: 
            self.get_logger().warn(f"Cannot freeze target. Invalid Depth: {self.food_depth}")
            return False
        
        bearing = self.pixel_to_bearing(self.food_u)
        abs_yaw = curr_yaw + bearing
        
        self.target_map_x = curr_x + self.food_depth * math.cos(abs_yaw)
        self.target_map_y = curr_y + self.food_depth * math.sin(abs_yaw)
        self.get_logger().info(f"Food Frozen at Map: ({self.target_map_x:.2f}, {self.target_map_y:.2f}) | Depth: {self.food_depth:.2f}")
        return True

    # [NEW] Helper function for continuous barking
    def perform_continuous_bark(self):
        now = self.get_clock().now()
        
        # 아직 한 번도 안 짖었거나, 지난번 짖은 후 충분한 시간이 지났다면
        should_bark = False
        if self.last_bark_time is None:
            should_bark = True
        else:
            elapsed = (now - self.last_bark_time).nanoseconds * 1e-9
            if elapsed > BARK_INTERVAL:
                should_bark = True
        
        if should_bark:
            msg = String()
            msg.data = "bark"
            self.voice_pub.publish(msg)
            # 로그는 너무 시끄러울 수 있으므로 디버그용으로만 남기거나 생략 가능
            # self.get_logger().info("Barking while approaching...")
            self.last_bark_time = now

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
            if self.is_detected_fresh():
                self.get_logger().info("Food Detected! Switching to CENTERING.")
                self.set_state("CENTERING")
                self.center_stable_cnt = 0
                self.patrol_finish_timer = None 
                
                # 발견 즉시 한 번 짖기 위해 초기화
                self.last_bark_time = None 
                self.perform_continuous_bark() 
                return
            
            if self.room_index >= len(ROOM_LIST):
                if self.patrol_finish_timer is None:
                    self.patrol_finish_timer = self.get_clock().now()
                    self.get_logger().info("End of path. Scanning for 5 seconds before finishing...")
                    return 

                elapsed_finish = (self.get_clock().now() - self.patrol_finish_timer).nanoseconds * 1e-9
                if elapsed_finish > 5.0:
                    self.get_logger().info("Patrol Finished. No food found.")
                    self.set_state("COMPLETE")
                return 

            target = ROOM_LIST[self.room_index]
            dist = math.hypot(target[1] - curr_x, target[2] - curr_y)
            
            self.publish_goal_smart(target[1], target[2], target[3])
            
            if dist < ARRIVAL_THRESHOLD:
                self.get_logger().info(f"Arrived at {target[0]}")
                self.room_index += 1

        # [STATE 2] CENTERING
        elif self.state == "CENTERING":
            # [NEW] Centering 중에도 흥분해서 짖음
            self.perform_continuous_bark()

            if not self.is_detected_fresh():
                self.get_logger().warn("Lost food during centering -> Back to PATROL")
                self.set_state("PATROL")
                return

            elapsed_state = (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9
            if elapsed_state > CENTERING_TIMEOUT_SEC:
                self.get_logger().warn("Centering Timeout! Trying to APPROACH directly or Next Room.")
                if self.food_depth > 0.1:
                    if self.freeze_food_position(curr_x, curr_y, curr_yaw):
                        self.set_state("APPROACH")
                    else:
                         self.set_state("PATROL")
                else:
                    self.set_state("PATROL")
                return

            center_u = self.get_image_center_u()
            
            if self.food_u <= 0:
                self.get_logger().info("Waiting for valid food_u coordinates...")
                return 

            u = self.food_u 
            diff_u = u - center_u
            
            bearing = self.pixel_to_bearing(u)
            target_yaw = curr_yaw + max(min(bearing, MAX_YAW_STEP), -MAX_YAW_STEP)
            
            self.publish_goal_smart(curr_x, curr_y, target_yaw)

            if abs(diff_u) < CENTER_TOL_PX and self.food_depth > 0.1:
                self.center_stable_cnt += 1
            else:
                self.center_stable_cnt = 0

            if self.center_stable_cnt >= CENTERING_STABLE_TICKS:
                if self.freeze_food_position(curr_x, curr_y, curr_yaw):
                    self.set_state("APPROACH")

        # [STATE 3] APPROACH
        elif self.state == "APPROACH":
            # [NEW] 다가가는 중에도 계속 짖음
            self.perform_continuous_bark()

            dist_to_food = math.hypot(self.target_map_x - curr_x, self.target_map_y - curr_y)
            
            if dist_to_food <= BARK_DISTANCE_THRESHOLD:
                self.get_logger().info("!!! FOUND FOOD (Target Reached) !!!")
                
                # 움직임 정지
                self.publish_goal_smart(curr_x, curr_y, curr_yaw)
                
                # [수정] 마지막으로 한 번 더 짖고 종료 (원한다면 생략 가능, 여기선 확실한 종료 알림용)
                self.perform_continuous_bark() 
                self.get_logger().info(">>> Mission Complete: Arrived at Food <<<")

                self.set_state("COMPLETE")
                return
            
            goal_yaw = math.atan2(self.target_map_y - curr_y, self.target_map_x - curr_x)
            self.publish_goal_smart(self.target_map_x, self.target_map_y, goal_yaw)

        # [STATE 4] COMPLETE
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
