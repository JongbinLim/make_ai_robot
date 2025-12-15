#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import numpy as np
import cv2
from ultralytics import YOLO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO("best.pt")

        # ---------------------------
        # Subscribers
        # ---------------------------
        self.rgb_sub = self.create_subscription(
            Image, "/camera_top/image", self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, "/camera_top/depth", self.depth_callback, 10)

        self.pc_sub = self.create_subscription(
            Image, "/camera_top/points", self.pc_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_top/camera_info", self.camera_info_callback, 10)

        # ---------------------------
        # Publishers
        # ---------------------------
        self.pub_det_image = self.create_publisher(Image, "/camera/detections/image", 10)
        
        # [Wide] 넓은 시야각 (20% ~ 80%)
        self.pub_labels = self.create_publisher(String, "/detections/labels", 10)
        
        # [Narrow] 좁은 시야각 (35% ~ 65%) - Code B 기능
        self.pub_narrow_labels = self.create_publisher(String, "/detections/narrow/labels", 10)

        self.pub_distance = self.create_publisher(Float32, "/detections/distance", 10)
        self.pub_speech = self.create_publisher(String, "/robot_dog/speech", 10)
        self.pub_stop_sign = self.create_publisher(Bool, "/perception/stop_sign", 10)

        # [Nurse] 간호사 위치 정보 - Code A 기능
        self.pub_nurse_center = self.create_publisher(PointStamped, "/detections/nurse_center", 10)

        # [NEW] 음식 위치 정보 (FoodSmartSearcher용)
        self.pub_food_center = self.create_publisher(PointStamped, "/detections/food_center", 10)

        # Storage
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        # Logic Variables
        self.edible_objects = {"apple", "banana", "pizza"}
        self.stop_labels = {"stop sign"}

        # Bark Logic (Code B)
        self.bark_distance_threshold = 0.4  # meters
        self.bark_cooldown_sec = 2.0
        self.last_bark_time = self.get_clock().now()

        self.get_logger().info("PerceptionNode initialized (Merged + Food Center).")

    # ---------------------------
    # Depth helper (From Code A)
    # ---------------------------
    def robust_depth(self, depth_img, x, y, k=9, q=30):
        """
        (x,y) 주변 kxk 패치에서 유효 depth(>0, finite)만 모아서
        '가까운 쪽' 퍼센타일(q)을 반환. (노이즈 제거 및 안전한 거리 측정)
        """
        h, w = depth_img.shape[:2]
        r = k // 2
        x1 = max(0, int(x) - r); x2 = min(w, int(x) + r + 1)
        y1 = max(0, int(y) - r); y2 = min(h, int(y) + r + 1)

        patch = depth_img[y1:y2, x1:x2].astype(np.float32)
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            return -1.0
        return float(np.percentile(vals, q))

    # ---------------------------
    # Callbacks
    # ---------------------------
    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run_detection()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def pc_callback(self, msg):
        pass

    def camera_info_callback(self, msg):
        self.camera_info = msg

    # ---------------------------
    # Object Detection + Publish
    # ---------------------------
    def run_detection(self):
        if self.rgb_image is None or self.depth_image is None:
            return

        img = self.rgb_image.copy()
        H, W = img.shape[:2]

        # ===== [Wide] 가로 중앙 60% 영역 (0.2W ~ 0.8W) =====
        wide_left = int(W * 0.2)
        wide_right = int(W * 0.8)

        # ===== [Narrow] 가로 중앙 30% 영역 (0.35W ~ 0.65W) =====
        narrow_left = int(W * 0.35)
        narrow_right = int(W * 0.65)

        # 디버깅용 선 그리기 (Wide: 노랑, Narrow: 파랑)
        cv2.line(img, (wide_left, 0), (wide_left, H), (0, 255, 255), 2)
        cv2.line(img, (wide_right, 0), (wide_right, H), (0, 255, 255), 2)
        cv2.line(img, (narrow_left, 0), (narrow_left, H), (255, 0, 0), 2)
        cv2.line(img, (narrow_right, 0), (narrow_right, H), (255, 0, 0), 2)

        results = self.model(img, verbose=False)[0]

        labels_wide = []
        labels_narrow = []
        
        distance_to_target = -1.0
        target_label = None
        
        stop_sign_detected = False
        bark_msg = "None"

        # Nurse 관련 변수
        nurse_found = False
        nurse_cx, nurse_cy, nurse_depth = -1, -1, -1.0
        best_nurse_depth = None  # 여러 명일 경우 가장 가까운 사람

        # [NEW] Food 관련 변수
        food_found = False
        food_cx, food_cy, food_depth = -1, -1, -1.0
        best_food_depth = None # 여러 음식 중 가장 가까운 것

        dh, dw = self.depth_image.shape[:2]

        for xyxy, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            label = results.names[int(cls)]

            # -----------------------------------------------
            # 1. Wide 영역 체크 (기본 로직)
            # -----------------------------------------------
            if not (wide_left <= cx <= wide_right):
                continue  # Wide 영역 밖이면 무시

            labels_wide.append(label)

            if label in self.stop_labels:
                stop_sign_detected = True

            # ---------- 일반 객체 거리 (Code A 방식: Robust Depth) ----------
            if 0 <= cx < dw and 0 <= cy < dh:
                depth_center = self.robust_depth(self.depth_image, cx, cy, k=7, q=50) # Median
            else:
                depth_center = -1.0

            # ---------- Nurse Logic ----------
            if label == "nurse":
                nurse_found = True
                sample_y = int(y1 + 0.8 * (y2 - y1))
                sample_x = cx

                if 0 <= sample_x < dw and 0 <= sample_y < dh:
                    depth_sample = self.robust_depth(self.depth_image, sample_x, sample_y, k=9, q=30)
                else:
                    depth_sample = -1.0

                if depth_sample <= 0 and depth_center > 0:
                    depth_sample = depth_center
                
                if depth_sample > 0:
                    if best_nurse_depth is None or depth_sample < best_nurse_depth:
                        best_nurse_depth = depth_sample
                        nurse_cx, nurse_cy, nurse_depth = cx, cy, depth_sample
                else:
                    if best_nurse_depth is None and (nurse_cx, nurse_cy) == (-1, -1):
                        nurse_cx, nurse_cy, nurse_depth = cx, cy, -1.0
                
                cv2.circle(img, (cx, int(y1 + 0.8 * (y2 - y1))), 5, (255, 0, 0), -1)

            # ---------- [NEW] Food Logic (for FoodSmartSearcher) ----------
            if label in self.edible_objects:
                food_found = True
                # 음식은 작으므로 그냥 Center Depth 사용 (depth_center)
                
                if depth_center > 0:
                    # 가장 가까운 음식 선택
                    if best_food_depth is None or depth_center < best_food_depth:
                        best_food_depth = depth_center
                        food_cx, food_cy, food_depth = cx, cy, depth_center
                else:
                    # Depth가 안 잡혀도 좌표는 저장 (Visual Servoing용)
                    if best_food_depth is None and (food_cx, food_cy) == (-1, -1):
                         food_cx, food_cy, food_depth = cx, cy, -1.0

                # 디버깅: 초록색 점
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

            # ---------- 전체 타겟 최소 거리 갱신 ----------
            if depth_center > 0 and (distance_to_target < 0 or depth_center < distance_to_target):
                distance_to_target = depth_center
                target_label = label

            # Draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # -----------------------------------------------
            # 2. Narrow 영역 체크 (Code B 기능)
            # -----------------------------------------------
            if narrow_left <= cx <= narrow_right:
                labels_narrow.append(label)

        # ---------------------------
        # Publish 1: Detection Image
        # ---------------------------
        det_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub_det_image.publish(det_msg)

        # ---------------------------
        # Publish 2: Labels (Wide & Narrow)
        # ---------------------------
        label_msg = String()
        label_msg.data = ", ".join(labels_wide)
        self.pub_labels.publish(label_msg)

        narrow_label_msg = String()
        narrow_label_msg.data = ", ".join(labels_narrow)
        self.pub_narrow_labels.publish(narrow_label_msg)

        # ---------------------------
        # Publish 3: Distance (Closest in Wide)
        # ---------------------------
        dist_msg = Float32()
        dist_msg.data = distance_to_target if distance_to_target > 0 else -1.0
        self.pub_distance.publish(dist_msg)

        # ---------------------------
        # Publish 4: Bark (with Cooldown - Code B)
        # ---------------------------
        now = self.get_clock().now()
        cooldown_ok = (now - self.last_bark_time).nanoseconds > int(self.bark_cooldown_sec * 1e9)

        if (cooldown_ok 
            and target_label is not None 
            and target_label in self.edible_objects 
            and distance_to_target > 0
            and distance_to_target <= self.bark_distance_threshold):
            
            bark_msg = "bark"
            self.last_bark_time = now

        bark = String()
        bark.data = bark_msg
        self.pub_speech.publish(bark)

        # ---------------------------
        # Publish 5: Stop Sign
        # ---------------------------
        stop_msg = Bool()
        stop_msg.data = stop_sign_detected
        self.pub_stop_sign.publish(stop_msg)

        # ---------------------------
        # Publish 6: Nurse Center (Code A)
        # ---------------------------
        nurse_msg = PointStamped()
        nurse_msg.header.stamp = self.get_clock().now().to_msg()
        if self.camera_info is not None and getattr(self.camera_info.header, "frame_id", ""):
            nurse_msg.header.frame_id = self.camera_info.header.frame_id
        else:
            nurse_msg.header.frame_id = "camera_top"

        if nurse_found:
            nurse_msg.point.x = float(nurse_cx)
            nurse_msg.point.y = float(nurse_cy)
            nurse_msg.point.z = float(nurse_depth if nurse_depth > 0 else -1.0)
        else:
            nurse_msg.point.x = -1.0
            nurse_msg.point.y = -1.0
            nurse_msg.point.z = -1.0
        
        self.pub_nurse_center.publish(nurse_msg)

        # ---------------------------
        # Publish 7: Food Center (NEW)
        # ---------------------------
        food_msg = PointStamped()
        food_msg.header = nurse_msg.header # 같은 헤더 사용

        if food_found:
            # FoodSmartSearcher에서 point.x를 u(pixel x)로, point.z를 depth로 사용
            food_msg.point.x = float(food_cx)
            food_msg.point.y = float(food_cy)
            food_msg.point.z = float(food_depth if food_depth > 0 else -1.0)
        else:
            food_msg.point.x = -1.0
            food_msg.point.y = -1.0
            food_msg.point.z = -1.0
            
        self.pub_food_center.publish(food_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
