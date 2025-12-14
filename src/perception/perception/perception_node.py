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

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/camera_top/image", self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, "/camera_top/depth", self.depth_callback, 10)

        self.pc_sub = self.create_subscription(
            Image, "/camera_top/points", self.pc_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_top/camera_info", self.camera_info_callback, 10)

        # Publishers
        self.pub_det_image = self.create_publisher(Image, "/camera/detections/image", 10)
        self.pub_labels = self.create_publisher(String, "/detections/labels", 10)
        self.pub_distance = self.create_publisher(Float32, "/detections/distance", 10)
        self.pub_speech = self.create_publisher(String, "/robot_dog/speech", 10)
        self.pub_stop_sign = self.create_publisher(Bool, "/perception/stop_sign", 10)

        self.pub_nurse_center = self.create_publisher(
            PointStamped, "/detections/nurse_center", 10
        )

        # storage
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        self.edible_objects = {"apple", "banana", "pizza"}
        self.stop_labels = {"stop sign"}

        self.get_logger().info("PerceptionNode initialized.")

    # ---------------------------
    # Depth helper (robust)
    # ---------------------------
    def robust_depth(self, depth_img, x, y, k=9, q=30):
        """
        (x,y) 주변 kxk 패치에서 유효 depth(>0, finite)만 모아서
        '가까운 쪽' 퍼센타일(q)을 반환.
        - q=50 => median
        - q=30 => background(멀리) 섞여도 가까운(사람)쪽을 더 잘 잡음
        없으면 -1
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

        # 중앙 60% 영역
        left_line = int(W * 0.2)
        right_line = int(W * 0.8)

        cv2.line(img, (left_line, 0), (left_line, H), (0, 255, 255), 2)
        cv2.line(img, (right_line, 0), (right_line, H), (0, 255, 255), 2)

        results = self.model(img, verbose=False)[0]

        labels = []
        distance_to_target = -1.0
        target_label = None
        bark_msg = "None"

        stop_sign_detected = False

        dh, dw = self.depth_image.shape[:2]

        # nurse publish
        nurse_found = False
        nurse_cx, nurse_cy, nurse_depth = -1, -1, -1.0
        best_nurse_depth = None  # 최소 depth(nurse 중 가장 가까움)

        for xyxy, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if not (left_line <= cx <= right_line):
                continue

            label = results.names[int(cls)]
            labels.append(label)

            if label in self.stop_labels:
                stop_sign_detected = True

            # ---------- 일반 객체 거리(센터 기반, median 권장) ----------
            if 0 <= cx < dw and 0 <= cy < dh:
                depth_center = self.robust_depth(self.depth_image, cx, cy, k=7, q=50)  # median
            else:
                depth_center = -1.0

            # ---------- nurse depth는 bbox 높이 0.8 지점에서 ----------
            if label == "nurse":
                nurse_found = True

                sample_y = int(y1 + 0.8 * (y2 - y1))  # ✅ bbox의 0.8 높이
                sample_x = cx

                if 0 <= sample_x < dw and 0 <= sample_y < dh:
                    depth_sample = self.robust_depth(self.depth_image, sample_x, sample_y, k=9, q=30)
                else:
                    depth_sample = -1.0

                # fallback: 샘플이 invalid면 센터 depth라도 사용
                if depth_sample <= 0 and depth_center > 0:
                    depth_sample = depth_center

                # "가장 가까운 nurse" 선택
                if depth_sample > 0:
                    if best_nurse_depth is None or depth_sample < best_nurse_depth:
                        best_nurse_depth = depth_sample
                        nurse_cx, nurse_cy, nurse_depth = cx, cy, depth_sample
                else:
                    if best_nurse_depth is None and (nurse_cx, nurse_cy) == (-1, -1):
                        nurse_cx, nurse_cy, nurse_depth = cx, cy, -1.0

            # ---------- 전체 타겟 최소 거리 갱신(여기는 기존처럼 센터 기준) ----------
            if depth_center > 0 and (distance_to_target < 0 or depth_center < distance_to_target):
                distance_to_target = depth_center
                target_label = label

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # (옵션) nurse depth 샘플 포인트 표시(디버깅)
            if label == "nurse":
                cv2.circle(img, (cx, int(y1 + 0.8 * (y2 - y1))), 5, (255, 0, 0), -1)

        # Publish detection image
        det_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub_det_image.publish(det_msg)

        # Publish labels
        label_msg = String()
        label_msg.data = ", ".join(labels)
        self.pub_labels.publish(label_msg)

        # Publish distance (closest object by center-depth)
        dist_msg = Float32()
        dist_msg.data = distance_to_target if distance_to_target > 0 else -1.0
        self.pub_distance.publish(dist_msg)

        # Publish bark / None
        if target_label is not None and target_label in self.edible_objects:
            bark_msg = "bark"

        bark = String()
        bark.data = bark_msg
        self.pub_speech.publish(bark)

        stop_msg = Bool()
        stop_msg.data = stop_sign_detected
        self.pub_stop_sign.publish(stop_msg)

        # Publish nurse center (x=cx, y=cy, z=depth_sample)
        nurse_msg = PointStamped()
        nurse_msg.header.stamp = self.get_clock().now().to_msg()
        if self.camera_info is not None and getattr(self.camera_info.header, "frame_id", ""):
            nurse_msg.header.frame_id = self.camera_info.header.frame_id
        else:
            nurse_msg.header.frame_id = "camera_top"

        if nurse_found:
            nurse_msg.point.x = float(nurse_cx)    # 이미지 u (center)
            nurse_msg.point.y = float(nurse_cy)    # 이미지 v (center)
            nurse_msg.point.z = float(nurse_depth if nurse_depth > 0 else -1.0)  # ✅ 0.8h depth
        else:
            nurse_msg.point.x = -1.0
            nurse_msg.point.y = -1.0
            nurse_msg.point.z = -1.0

        self.pub_nurse_center.publish(nurse_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
