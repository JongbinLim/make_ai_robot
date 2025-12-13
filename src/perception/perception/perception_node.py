#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge

import numpy as np
import cv2

# YOLO
from ultralytics import YOLO


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        ### ----------------------------
        # Load YOLO model
        ### ----------------------------
        self.model = YOLO("best.pt")

        ### ----------------------------
        # Subscribers
        ### ----------------------------
        self.rgb_sub = self.create_subscription(
            Image, "/camera_top/image", self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, "/camera_top/depth", self.depth_callback, 10)

        self.pc_sub = self.create_subscription(
            Image, "/camera_top/points", self.pc_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_top/camera_info", self.camera_info_callback, 10)

        ### ----------------------------
        # Publishers
        ### ----------------------------
        self.pub_det_image = self.create_publisher(Image, "/camera/detections/image", 10)
        self.pub_labels = self.create_publisher(String, "/detections/labels", 10)
        self.pub_distance = self.create_publisher(Float32, "/detections/distance", 10)
        self.pub_speech = self.create_publisher(String, "/robot_dog/speech", 10)

        # 저장 공간
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        # 먹을 수 있는 물체(예시)
        self.edible_objects = {"apple", "banana", "orange", "cake"}

        self.get_logger().info("PerceptionNode initialized.")

    # ---------------------------
    # Callbacks
    # ---------------------------
    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run_detection()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def pc_callback(self, msg):
        # point cloud는 선택적 사용 → 필요하면 추가 처리
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

        # ===== 가로 중앙 60% 영역(0.2W ~ 0.8W) ===== 더 좁게 설정 가능 -> 0.3 ~ 0.7 등
        left_line = int(W * 0.2)
        right_line = int(W * 0.8)

        # (선택) 디버깅용: 중앙 영역 기준선 그리기
        cv2.line(img, (left_line, 0), (left_line, H), (0, 255, 255), 2)
        cv2.line(img, (right_line, 0), (right_line, H), (0, 255, 255), 2)

        # YOLO inference
        results = self.model(img, verbose=False)[0]

        labels = []
        distance_to_target = -1.0
        target_label = None
        bark_msg = "None"

        # depth 이미지 크기
        dh, dw = self.depth_image.shape[:2]

        for xyxy, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            # 중심점 계산
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ===== 핵심: 가로 중앙 60% 안에 들어온 것만 "인식" =====
            if not (left_line <= cx <= right_line):
                continue

            label = results.names[int(cls)]
            labels.append(label)

            # depth 범위 체크
            if 0 <= cx < dw and 0 <= cy < dh:
                depth = float(self.depth_image[cy, cx])
            else:
                depth = -1.0

            # 최소 distance 갱신 (유효 depth만)
            if depth > 0 and (distance_to_target < 0 or depth < distance_to_target):
                distance_to_target = depth
                target_label = label

            # bounding box 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        # ---------------------------
        # Publish detection image
        # ---------------------------
        det_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub_det_image.publish(det_msg)

        # ---------------------------
        # Publish labels
        # ---------------------------
        label_msg = String()
        label_msg.data = ", ".join(labels)
        self.pub_labels.publish(label_msg)

        # ---------------------------
        # Publish distance
        # ---------------------------
        dist_msg = Float32()
        dist_msg.data = distance_to_target if distance_to_target > 0 else -1.0
        self.pub_distance.publish(dist_msg)

        # ---------------------------
        # Publish bark / None
        # ---------------------------
        # 중앙 60% 안에서 잡힌 "가장 가까운" 물체가 edible이면 bark
        if target_label is not None and target_label in self.edible_objects:
            bark_msg = "bark"

        bark = String()
        bark.data = bark_msg
        self.pub_speech.publish(bark)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
