#!/usr/bin/env python3

"""
ROS2 global localizer node (Complete Version)

This node implements Monte Carlo Localization (MCL) using a Particle Filter.
It subscribes to map, scan, and tf to localize the robot in the map frame.

Flow:
1. Initialize Particle Filter with given parameters.
2. Receive Map -> PF.set_map()
3. Receive Scan:
    a. Calculate motion delta from Odom TF (Prediction) -> PF.predict()
    b. Update particles with LaserScan (Correction) -> PF.update()
    c. Resample particles -> PF.resample()
    d. Calculate Map->Odom transform
    e. Publish Map->Odom TF and visualization topics
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
import math

from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, PoseArray, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, \
    ExtrapolationException
import tf_transformations
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np

# 사용자 정의 모듈 임포트
from utils import pose_to_matrix, transform_to_matrix
from particle_filter_unoptimized import ParticleFilter


class GlobalLocalizerNode(Node):
    def __init__(self):
        super().__init__('global_localizer')
        self.get_logger().info('Global localizer (MCL) node initialized')

        # Parameters
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 1.0)  # Default y=1
        self.declare_parameter('initial_pose_z', 0.5)  # Default z=0.5
        self.declare_parameter('initial_pose_roll', 0.0)
        self.declare_parameter('initial_pose_pitch', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)
        self.declare_parameter('min_particles', 800)
        self.declare_parameter('max_particles', 3000)
        self.declare_parameter('max_position_jump', 0.5)  # 미터 단위
        self.declare_parameter('max_yaw_jump', 0.5)  # 라디안 단위 (약 28도)

        self.init_x = self.get_parameter('initial_pose_x').value
        self.init_y = self.get_parameter('initial_pose_y').value
        self.init_z = self.get_parameter('initial_pose_z').value  # 저장 필요
        init_yaw = self.get_parameter('initial_pose_yaw').value

        self.max_pos_jump_sq = self.get_parameter('max_position_jump').value ** 2  # 거리 비교시 제곱 사용 (성능 최적화)
        self.max_yaw_jump = self.get_parameter('max_yaw_jump').value

        # Particle Filter Initialization
        self.pf = ParticleFilter(
            min_particles=self.get_parameter('min_particles').value,
            max_particles=self.get_parameter('max_particles').value,
            initial_noise=[0.1, 0.1, 0.1]  # x, y, yaw noise
        )
        # 초기 파티클 생성
        self.pf.initialize(self.init_x, self.init_y, init_yaw)

        self.map_received = False
        self.last_odom_matrix = None
        self.last_scan_time = None
        self.last_valid_T_map_to_odom = None

        # TF Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Map Subscription
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # Initial Pose Subscription (RViz "2D Pose Estimate")
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            10
        )

        # Laser Scan Subscription (Main Loop)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Visualization Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/go1_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)

    def map_callback(self, msg):
        """지도를 수신하면 파티클 필터에 설정합니다."""
        self.get_logger().info(f'Received map: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution}')
        self.pf.set_map(msg)
        self.map_received = True

    def initial_pose_callback(self, msg):
        """RViz 등에서 초기 위치를 재설정할 때 호출됩니다."""
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Quaternion to Euler (Yaw)
        q = msg.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.get_logger().info(f'Relocalizing to x:{x:.2f}, y:{y:.2f}, yaw:{yaw:.2f}')
        self.pf.initialize(x, y, yaw)
        self.last_odom_matrix = None  # 오도메트리 연속성 초기화

    def get_odom_pose(self, time):
        """특정 시간의 odom -> base_link TF를 구합니다."""
        try:
            # 1. 정확한 시간 시도 (timeout 인자는 rclpy 버전에 따라 다를 수 있음, 보통 loop 내에서 체크)
            # 하지만 blocking을 피하기 위해 예외 처리 방식으로 접근합니다.
            trans = self.tf_buffer.lookup_transform('odom', 'base', time)
            return transform_to_matrix(trans.transform)
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                # 2. 실패 시 가장 최신 시간 조회 (0)
                # 단, 시간 차이가 너무 크면(예: 0.2초 이상) 무시하는 로직 추가 권장
                trans = self.tf_buffer.lookup_transform('odom', 'base', Time())
                # 로그: self.get_logger().warn('Using latest TF instead of synced timestamp')
                return transform_to_matrix(trans.transform)
            except Exception as e:
                self.get_logger().debug(f'TF lookup completely failed: {e}')
                return None

    def scan_callback(self, scan_msg):
        """
        메인 로직:
        1. Prediction (Odom Delta)
        2. Update (Likelihood Field)
        3. Resample
        4. TF Publish
        """
        if not self.map_received:
            return

        # 현재 스캔 시간
        current_time = Time.from_msg(scan_msg.header.stamp)

        # 1. Get Current Odom Pose
        curr_odom_matrix = self.get_odom_pose(current_time)
        if curr_odom_matrix is None:
            return

        # 2. Prediction Step (Motion Model)
        if self.last_odom_matrix is not None:
            # 이전 프레임 기준 상대 변환 행렬 계산: T_prev_inv * T_curr
            # 이것은 Global(Odom) frame에서의 이동이므로, Robot frame에서의 dx, dy로 변환해야 함

            # 로봇 좌표계 기준 이동량 계산
            # T_delta = inv(T_prev) @ T_curr
            # T_delta는 로봇이 '이전 위치'에서 보았을 때 '현재 위치'가 어디인지 나타냄
            T_prev_inv = np.linalg.inv(self.last_odom_matrix)
            T_delta = T_prev_inv @ curr_odom_matrix

            dx = T_delta[0, 3]
            dy = T_delta[1, 3]

            # 회전량 계산 (rotation matrix to euler)
            # T_delta[:3, :3] 은 회전 행렬
            _, _, dyaw = tf_transformations.euler_from_matrix(T_delta)

            # 이동량이 매우 작으면 필터 업데이트 스킵
            if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dyaw) < 0.001:
                return  # 업데이트 하지 않음

            # 파티클 필터 예측 단계 실행
            self.pf.predict(dx, dy, dyaw)

        self.last_odom_matrix = curr_odom_matrix

        # 3. Update Step (Sensor Model)
        # LaserScan to Ranges handling
        # PF expects specific arguments.
        self.pf.update(
            scan_msg.ranges,
            scan_msg.angle_min,
            scan_msg.angle_increment,
            sensor_offset=[0.0, 0.0]  # base_link와 laser_link가 일치한다고 가정. 다르면 TF로 오프셋 계산 필요.
        )

        # 4. Get Estimated Pose (Map -> Base)
        pf_est_pose = self.pf.get_estimated_pose()  # [x, y, yaw]

        # 5. Calculate Map -> Odom TF
        # T_map_to_base = T_map_to_odom * T_odom_to_base
        # => T_map_to_odom = T_map_to_base * inv(T_odom_to_base)

        # T_map_to_base 생성
        q_est = tf_transformations.quaternion_from_euler(0, 0, pf_est_pose[2])
        T_map_to_base = pose_to_matrix([
            pf_est_pose[0], pf_est_pose[1], 0.0,
            q_est[0], q_est[1], q_est[2], q_est[3]
        ])

        # T_odom_to_base의 역행렬
        T_base_to_odom = np.linalg.inv(curr_odom_matrix)

        # T_map_to_odom 계산
        T_map_to_odom_candidate = T_map_to_base @ T_base_to_odom

        # --- 4. Safety Check (Position & Yaw Divergence) ---
        final_T_map_to_odom = T_map_to_odom_candidate

        if self.last_valid_T_map_to_odom is not None:
            # A. Position Check
            prev_trans = self.last_valid_T_map_to_odom[:3, 3]
            curr_trans = T_map_to_odom_candidate[:3, 3]
            # 제곱 거리로 비교 (sqrt 연산 절약)
            dist_sq = np.sum((curr_trans - prev_trans) ** 2)

            # B. Yaw Check
            _, _, prev_map_yaw = tf_transformations.euler_from_matrix(self.last_valid_T_map_to_odom)
            _, _, curr_map_yaw = tf_transformations.euler_from_matrix(T_map_to_odom_candidate)

            # 각도 차이 정규화 (atan2 사용: -pi ~ +pi 범위로 변환)
            yaw_diff = curr_map_yaw - prev_map_yaw
            yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))

            # C. Divergence Condition
            is_pos_jump = dist_sq > self.max_pos_jump_sq
            is_yaw_jump = abs(yaw_diff) > self.max_yaw_jump

            if is_pos_jump or is_yaw_jump:
                jump_type = []
                if is_pos_jump: jump_type.append(f"Dist {math.sqrt(dist_sq):.2f}m")
                if is_yaw_jump: jump_type.append(f"Yaw {math.degrees(yaw_diff):.2f}deg")

                self.get_logger().warn(f"⚠️ Divergence Detected ({', '.join(jump_type)}). Resetting to Odom.")

                # [Reset Logic]
                # 1. 이전 안전 변환 사용
                final_T_map_to_odom = self.last_valid_T_map_to_odom

                # 2. 안전한 현재 로봇 위치 역산
                T_map_to_base_safe = final_T_map_to_odom @ curr_odom_matrix
                safe_x = T_map_to_base_safe[0, 3]
                safe_y = T_map_to_base_safe[1, 3]
                _, _, safe_yaw = tf_transformations.euler_from_matrix(T_map_to_base_safe)

                # 3. 파티클 강제 리셋
                self.pf.initialize(safe_x, safe_y, safe_yaw)
                pf_est_pose = [safe_x, safe_y, safe_yaw]
            else:
                self.last_valid_T_map_to_odom = T_map_to_odom_candidate
        else:
            self.last_valid_T_map_to_odom = T_map_to_odom_candidate

        # 5. Publish
        self.publish_tf(final_T_map_to_odom, scan_msg.header.stamp)
        self.publish_mcl_pose(pf_est_pose, scan_msg.header.stamp)
        self.publish_particles(scan_msg.header.stamp)

    def publish_tf(self, T, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        translation = T[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(T)

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_mcl_pose(self, pose_2d, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(pose_2d[0])
        msg.pose.position.y = float(pose_2d[1])
        msg.pose.position.z = self.init_z # Go1은 평지 보행 가정 시 z값 고정 혹은 IMU/Odom에서 가져와야 함. 일단 초기값 사용.

        # 2D PF는 yaw만 추정하므로, roll/pitch는 0 혹은 IMU 값 사용해야 함.
        # 지침상 단순화를 위해 0 혹은 초기값 유지.
        q = tf_transformations.quaternion_from_euler(0, 0, float(pose_2d[2]))

        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.pose_pub.publish(msg)

    def publish_particles(self, stamp):
        # 성능을 위해 파티클을 다운샘플링하여 시각화할 수도 있음
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'

        # Numba 배열이므로 numpy로 처리
        particles = self.pf.particles
        # 시각화 부하를 줄이기 위해 최대 100개 정도만 퍼블리시 하거나 전체를 퍼블리시
        step = max(1, len(particles) // 100)

        for p in particles[::step]:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            q = tf_transformations.quaternion_from_euler(0, 0, float(p[2]))
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)

        self.particle_pub.publish(msg)


if __name__ == '__main__':
    rclpy.init()
    node = GlobalLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
