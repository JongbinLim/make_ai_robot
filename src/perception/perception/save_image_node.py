#!/usr/bin/env python3

import sys
import termios
import tty
import threading
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

SAVE_ROOT = "/home/ros/craip_ws/save_images"


class SaveImageNode(Node):
    def __init__(self):
        super().__init__("save_image_node")

        self.bridge = CvBridge()
        self.is_saving = False
        self.rgb_image = None
        self.frame_id = 0

        # 세션 폴더 자동 생성
        self.session_dir = self.create_new_session_dir()

        # subscriber
        self.sub = self.create_subscription(
            Image,
            "/camera_top/image",
            self.image_callback,
            10
        )

        self.get_logger().info("SaveImageNode started")
        print(f"Saving folder: {self.session_dir}")
        print("Press 's' = start, 'e' = end, 'q' = quit")

        # 🔥 키 입력 스레드 시작
        self.key_thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self.key_thread.start()

    # ---------------------------
    def create_new_session_dir(self):
        os.makedirs(SAVE_ROOT, exist_ok=True)
        existing = [d for d in os.listdir(SAVE_ROOT) if d.isdigit()]

        num = (max(int(x) for x in existing) + 1) if existing else 0
        path = os.path.join(SAVE_ROOT, str(num))
        os.makedirs(path, exist_ok=True)
        return path

    # ---------------------------
    def image_callback(self, msg):
        if not self.is_saving:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        filename = os.path.join(self.session_dir, f"frame_{self.frame_id:06d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        self.frame_id += 1

    # ---------------------------
    # 기존 blocking 함수 그대로 사용
    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return key

    # ---------------------------
    # 🔥 별도 스레드에서 키 입력 감지
    def keyboard_thread(self):
        while True:
            key = self.get_key()

            if key == 's':
                self.is_saving = True
                print("\n=== START SAVING ===")

            elif key == 'e':
                self.is_saving = False
                print("\n=== STOP SAVING ===")

            elif key == 'q':
                print("\n=== QUIT ===")
                os._exit(0)

    # ---------------------------
    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)


def main(args=None):
    rclpy.init(args=args)
    node = SaveImageNode()
    node.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
