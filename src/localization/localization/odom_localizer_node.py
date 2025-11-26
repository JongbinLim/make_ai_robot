#!/usr/bin/env python3

"""
ROS2 odom localizer node

This node is responsible for localizing the robot in the odom frame.
It uses various sensors to localize the robot in the odom frame.
To overcome the limitations of drift, noise, and other factors, global localization is used to correct the odom localization.

odom_localizer: tf from odom to base_link frame (TF를 확인해 보니 base_link가 아니라 base로 되어있었음. 주의)
global_localizer: tf from map to odom frame
combined: tf from map to base_link frame (TF를 확인해 보니 base_link가 아니라 base로 되어있었음. 주의)

Current code uses Iterative Closest Point (ICP) to localize the robot in the odom frame.
You can modify this node to use other sensors to localize the robot.
Usually, LiDAR is used to localize the robot in the map frame, for global localization with given map. 
Odom is usually done by cmd_vel, IMU, RGBD camera, etc. 
But for simplicity, we use ICP to localize the robot in the odom frame.
"""

""" 시간 되면 UKF로 sensor fusion 구현하기"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan, Imu
from tf2_ros import TransformBroadcaster
import tf_transformations
import numpy as np
from utils import scan_to_pcd, icp_2d

def normalize_angle(angle):
    """각도를를 -pi ~ pi범위로 정규화 시켜줍니다."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class OdomLocalizerNode(Node):
    def __init__(self):
        # Initialize the ROS2 node
        super().__init__('odom_localizer')
        self.get_logger().info('Odom localizer node initialized')

        # Parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('max_jump_dist', 0.5)  # 한 번에 허용할 최대 이동 거리 (m)
        self.declare_parameter('max_jump_angle', 0.5)  # 한 번에 허용할 최대 회전 각도 (ra)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.max_jump_dist = self.get_parameter('max_jump_dist').value
        self.max_jump_angle = self.get_parameter('max_jump_angle').value

        # QoS Settings (센서 데이터 유실 방지 및 호환성)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10) # 라이다 정보
        self.imu_sub = self.create_subscription(Imu, '/imu_plugin/out', self.imu_callback, 10)

        # Create a tf publisher
        self.tf_broadcaster = TransformBroadcaster(self)

        # ICP를 위한 정보 저장
        self.previous_pcd = None
        self.current_pcd = None

        # IMU 변수들
        self.current_imu_yaw = 0.0  # IMU에서 적분된 현재 Yaw (Global 누적)
        self.last_imu_time = None
        self.keyframe_imu_yaw = 0.0  # Keyframe이 생성된 시점의 IMU Yaw

        # ICP 파라미터 값들 (수정 가능)
        self.max_iterations = 20
        self.tolerance = 1e-4
        self.distance_threshold = 0.2

        # Current pose in the odom frame ??
        self.current_pose = np.eye(3)
        # 마지막으로 pcd를 갱신했을 때의 global pose
        self.last_keyframe_pose = np.eye(3)

    def scan_callback(self, msg):
        """LaserScan Callback: ICP 수행 및 Pose 업데이트"""
        start_time = self.get_clock().now()

        # Scan data를 PCD로 변환
        new_pcd = scan_to_pcd(msg)

        if new_pcd is None or len(new_pcd) < 10:
            return

        self.current_pcd = new_pcd

        # Initialization
        if self.previous_pcd is None:
            self.previous_pcd = self.current_pcd
            self.keyframe_imu_yaw = self.current_imu_yaw  # 초기 Keyframe의 IMU 각도 저장
            self.last_keyframe_pose = np.eye(3)
            self.current_pose = np.eye(3)
            self.publish_tf(msg.header.stamp)
            return

        # IMU에 기반하여 현재 스캔 시점과 마지막 Keyframe 시점 사이의 회전량 계산
        delta_yaw_imu = normalize_angle(self.current_imu_yaw - self.keyframe_imu_yaw)

        # Pre-rotation: IMU 회전만큼 포인트를 미리 회전시켜 ICP의 부담을 줄임 (안정성 향상)
        # 이렇게 하면 ICP는 Rotation은 거의 0에 가깝고 Translation만 찾으면 됩니다.
        c_pred = np.cos(delta_yaw_imu)
        s_pred = np.sin(delta_yaw_imu)
        # Pre-rotation Matrix (2x2)
        rotation_matrix_pred = np.array([[c_pred, -s_pred], [s_pred, c_pred]])

        # 현재 스캔을 IMU 예측만큼 회전시킴
        rotated_pcd = self.current_pcd @ rotation_matrix_pred.T

        try:
            # Pre-rotation을 한 후의 잔차를 계산
            correction_mtx = icp_2d(
                previous_pcd=self.previous_pcd,
                current_pcd=rotated_pcd,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                distance_threshold=self.distance_threshold
            )
        except Exception:
            # ICP 실패 시 IMU 값만이라도 사용하여 업데이트 시도
            correction_mtx = np.eye(3)

        # Transform 융합 (Fusion)
        # T_pred: IMU로 예측한 회전 변환 (3x3)
        # Total Transform = IMU_Rotation * ICP_Correction
        T_pred = np.eye(3)
        T_pred[:2, :2] = rotation_matrix_pred

        # 최종 변환 행렬: Keyframe -> Current
        # T_final = T_pred @ correction_matrix (먼저 IMU만큼 돌리고, 그 다음 ICP 보정)
        T_local = correction_mtx @ T_pred

        # 추출
        dx = T_local[0, 2]
        dy = T_local[1, 2]
        dtheta = np.arctan2(T_local[1, 0], T_local[0, 0])

        # 안전장치. 갑작스럽게 값이 튀는 물리적으로 불가능한 상황을 예방
        if (np.sqrt(dx ** 2 + dy ** 2) > self.max_jump_dist) or (abs(dtheta) > self.max_jump_angle):
            # 급격한 튐 발생 시 업데이트 건너뜀
            self.publish_tf(msg.header.stamp)
            return

        # Global Pose 업데이트
        # Current = Keyframe * Local_Transform
        self.current_pose = self.last_keyframe_pose @ T_local

        # 마지막 Keyframe 대비 현재 위치의 변화량 계산
        dist_moved = dx ** 2 + dy ** 2
        rot_moved = abs(dtheta)

        # 이동량이 충분할 때만 기준 PCD 갱신 (구체적 수치는 수정 가능)
        if dist_moved > 0.02 or rot_moved > 0.05:
            self.previous_pcd = self.current_pcd # 다음 비교를 위해 Raw Data 저장
            self.last_keyframe_pose = self.current_pose
            self.keyframe_imu_yaw = self.current_imu_yaw  # Keyframe 갱신 시점의 IMU 각도 저장

        # Publish TF
        self.publish_tf(msg.header.stamp)

        # Frequency logging
        end_time = self.get_clock().now()
        time_delta = (end_time - start_time).nanoseconds / 1e9
        frequency = 1.0 / time_delta
        self.get_logger().info(f'ICP registration frequency: {frequency:.3f} Hz')

    def imu_callback(self, msg):
        """
        IMU 데이터로부터 각속도(yaw rate)를 적분하여 현재의 회전각을 추적합니다.
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = current_time
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        # 각속도(angular_velocity.z) 적분 -> Yaw 계산
        self.current_imu_yaw += msg.angular_velocity.z * dt
        self.current_imu_yaw = normalize_angle(self.current_imu_yaw)

    def publish_tf(self, timestamp):
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = float(self.current_pose[0, 2])
        t.transform.translation.y = float(self.current_pose[1, 2])
        t.transform.translation.z = 0.0

        yaw = np.arctan2(self.current_pose[1, 0], self.current_pose[0, 0])
        q = tf_transformations.quaternion_from_euler(0, 0, float(yaw))

        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)


if __name__ == '__main__':
    rclpy.init()
    node = OdomLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
