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
        self.model = YOLO("best.pt")

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/camera_top/image", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera_top/depth", self.depth_callback, 10)
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

        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        self.edible_objects = {"apple", "banana", "pizza"}
        self.stop_labels = {"stop sign"}

        self.get_logger().info("PerceptionNode initialized.")

    # ---------------------------
    # Depth helper
    # ---------------------------
    def robust_depth(self, depth_img, x, y, k=7, q=50):
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
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    # ---------------------------
    # Detection
    # ---------------------------
    def run_detection(self):
        if self.rgb_image is None or self.depth_image is None:
            return

        img = self.rgb_image.copy()
        H, W = img.shape[:2]

        left_line = int(W * 0.2)
        right_line = int(W * 0.8)

        results = self.model(img, verbose=False)[0]

        labels = []
        distance_to_target = -1.0
        bark_triggered = False
        stop_sign_detected = False

        dh, dw = self.depth_image.shape[:2]

        nurse_found = False
        nurse_cx = nurse_cy = nurse_depth = -1
        best_nurse_depth = None

        for xyxy, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if not (left_line <= cx <= right_line):
                continue

            label = results.names[int(cls)]
            if label in {"suitcase"}:
                label = "box"
            labels.append(label)

            if label in self.stop_labels:
                stop_sign_detected = True

            # ---------------------------
            # depth selection per class
            # ---------------------------
            if label in {"banana", "pizza", "nurse"}:
                if label in {"banana", "pizza"}:
                    sample_y = int(y1 + 0.7 * (y2 - y1))  # ✅ 0.3 height
                if label in {"nurse"}:
                    sample_y = int(y1 + 0.3 * (y2 - y1))
                if 0 <= cx < dw and 0 <= sample_y < dh:
                    depth_used = self.robust_depth(self.depth_image, cx, sample_y)
                else:
                    depth_used = -1.0


            else:
                if 0 <= cx < dw and 0 <= cy < dh:
                    depth_used = self.robust_depth(self.depth_image, cx, cy)
                else:
                    depth_used = -1.0

            # bark condition
            if label in self.edible_objects and 0 < depth_used <= 3.0:
                bark_triggered = True

            # nurse special handling
            if label == "nurse":
                nurse_found = True
                sy = int(y1 + 0.8 * (y2 - y1))
                if 0 <= cx < dw and 0 <= sy < dh:
                    d = self.robust_depth(self.depth_image, cx, sy, k=9, q=30)
                    if d > 0 and (best_nurse_depth is None or d < best_nurse_depth):
                        best_nurse_depth = d
                        nurse_cx, nurse_cy, nurse_depth = cx, cy, d


            # closest object
            if depth_used > 0 and (distance_to_target < 0 or depth_used < distance_to_target):
                distance_to_target = depth_used

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.pub_det_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        self.pub_labels.publish(String(data=", ".join(labels)))
        self.pub_distance.publish(
            Float32(data=distance_to_target if distance_to_target > 0 else -1.0)
        )
        self.pub_speech.publish(String(data="bark" if bark_triggered else "None"))
        self.pub_stop_sign.publish(Bool(data=stop_sign_detected))

        nurse_msg = PointStamped()
        nurse_msg.header.stamp = self.get_clock().now().to_msg()
        nurse_msg.header.frame_id = (
            self.camera_info.header.frame_id if self.camera_info else "camera_top"
        )

        if nurse_found:
            nurse_msg.point.x = float(nurse_cx)
            nurse_msg.point.y = float(nurse_cy)
            nurse_msg.point.z = float(nurse_depth)
        else:
            nurse_msg.point.x = nurse_msg.point.y = nurse_msg.point.z = -1.0

        self.pub_nurse_center.publish(nurse_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
