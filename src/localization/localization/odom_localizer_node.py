#!/usr/bin/env python3

"""
ROS2 Odom Localizer Node with Robust UKF Fusion & Pre-integration
[Version 2.0]
- Refined Transform Composition (IMU Guess + ICP Correction)
- Adaptive Measurement Noise based on Motion
- Minimized Lock Contention
- Robust Initialization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan, Imu
from tf2_ros import TransformBroadcaster
import tf_transformations

import numpy as np
from collections import deque
from bisect import bisect_left
from threading import Lock
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# --- Placeholder for ICP ---
# 실제 사용 시에는 Open3D나 별도의 최적화된 모듈 사용 권장
from utils import icp_2d


# --- Helper Functions ---
def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quaternion_from_euler(roll, pitch, yaw):
    return tf_transformations.quaternion_from_euler(roll, pitch, yaw)


# --- UKF Class ---
class RobotUKF:
    def __init__(self, dt=0.01, Q_params=None, R_icp_params=None, R_imu_params=None):
        # Sigma Points
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0)

        self.ukf = UnscentedKalmanFilter(
            dim_x=5, dim_z=3, dt=dt, fx=self.fx, hx=self.hx, points=points
        )

        # State: [x, y, theta, v, omega]
        self.ukf.x = np.zeros(5)
        self.ukf.P = np.eye(5) * 0.1

        # Process Noise (Q)
        q_diag = Q_params if Q_params else [0.001, 0.001, 0.001, 0.1, 0.1]
        self.ukf.Q = np.diag(q_diag)
        self.base_Q = np.diag(q_diag)

        # Measurement Noise (R)
        r_icp_diag = R_icp_params if R_icp_params else [0.01, 0.01, 0.005]
        self.R_icp_base = np.diag(r_icp_diag)
        self.R_icp = self.R_icp_base.copy()

        r_imu_diag = R_imu_params if R_imu_params else [0.01]
        self.R_imu = np.diag(r_imu_diag)

        # --- [수정된 부분] ---
        # x_mean과 z_mean에는 '함수 이름'을 할당해야 합니다.
        self.ukf.x_mean = self.state_mean
        self.ukf.z_mean = self.measurement_mean

        # 잔차 계산 함수 등록
        self.ukf.residual_x = self.residual_x
        self.ukf.residual_z = self.residual_h

    # --- [새로 추가/수정해야 할 함수들] ---

    def state_mean(self, sigmas, Wm):
        x = np.zeros(5)
        # 선형 변수 (x, y, v, omega)
        x[0] = np.dot(sigmas[:, 0], Wm)
        x[1] = np.dot(sigmas[:, 1], Wm)
        x[3] = np.dot(sigmas[:, 3], Wm)
        x[4] = np.dot(sigmas[:, 4], Wm)

        # 각도 변수 (theta) : 잔차 평균법
        ref_angle = sigmas[0, 2]
        diff_angles = sigmas[:, 2] - ref_angle

        # 각도 정규화 (-pi ~ pi)
        diff_angles = (diff_angles + np.pi) % (2 * np.pi) - np.pi

        avg_diff = np.dot(diff_angles, Wm)
        x[2] = (ref_angle + avg_diff + np.pi) % (2 * np.pi) - np.pi

        return x

    def measurement_mean(self, sigmas, Wm):
        """
        측정 벡터의 평균을 계산하는 함수 (잔차 평균법 적용)
        """
        dim_z = sigmas.shape[1]
        z = np.zeros(dim_z)

        # Case 1: ICP Update (z = [x, y, theta])
        # 차원이 3인 경우, 마지막 성분이 각도(Theta)라고 가정합니다.
        if dim_z == 3:
            # 1. 선형 변수 (x, y) -> 일반 가중 평균
            z[0] = np.dot(sigmas[:, 0], Wm)
            z[1] = np.dot(sigmas[:, 1], Wm)

            # 2. 각도 변수 (theta) -> 잔차 평균법 (Residual Mean)
            # 첫 번째 시그마 포인트의 각도를 기준(Reference)으로 삼습니다.
            ref_angle = sigmas[0, 2]

            # 모든 시그마 포인트와 기준 각도의 차이를 구합니다.
            diff_angles = sigmas[:, 2] - ref_angle

            # [중요] 각도 차이를 -pi ~ pi 사이로 정규화합니다.
            diff_angles = (diff_angles + np.pi) % (2 * np.pi) - np.pi

            # 차이값들의 가중 평균을 구합니다.
            avg_diff = np.dot(diff_angles, Wm)

            # 기준 각도에 평균 차이를 더해 최종 평균을 구합니다.
            z[2] = ref_angle + avg_diff

            # 최종 결과도 정규화해줍니다.
            z[2] = (z[2] + np.pi) % (2 * np.pi) - np.pi

        # Case 2: IMU Update (z = [omega])
        # 각속도(omega)는 각도가 아니라 '값'이므로 일반 평균을 씁니다.
        else:
            z[0] = np.dot(sigmas[:, 0], Wm)

        return z

    # (나머지 fx, hx, residual_x, residual_h 함수들은 기존 유지)
    def fx(self, x, dt):
        # ... (기존 코드와 동일)
        theta, v, omega = x[2], x[3], x[4]
        if abs(omega) > 1e-5:
            s_t = np.sin(theta)
            c_t = np.cos(theta)
            s_t_next = np.sin(theta + omega * dt)
            c_t_next = np.cos(theta + omega * dt)
            next_x = x[0] + (v / omega) * (s_t_next - s_t)
            next_y = x[1] + (v / omega) * (-c_t_next + c_t)
        else:
            next_x = x[0] + v * np.cos(theta + omega * dt / 2) * dt
            next_y = x[1] + v * np.sin(theta + omega * dt / 2) * dt

        next_theta = normalize_angle(theta + omega * dt)
        next_v = v * 0.95
        next_omega = omega * 0.90
        return np.array([next_x, next_y, next_theta, next_v, next_omega])

    def hx(self, x):
        return np.array([x[0], x[1], x[2]])

    def hx_imu(self, x):
        return np.array([x[4]])

    def residual_x(self, a, b):
        y = a - b
        y[2] = normalize_angle(y[2])
        return y

    def residual_h(self, a, b):
        y = a - b
        # ICP 측정값(3차원)일 때만 각도 정규화 수행
        if len(y) == 3:
            y[2] = normalize_angle(y[2])
        return y

    def angle_mean(self, sigmas, Wm):
        s = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        c = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        return np.arctan2(s, c)

    def predict(self, dt):
        # [수정됨] 내부 필터 객체(self.ukf)의 P 행렬에 접근하여 안정화 수행
        # 1. 대칭화
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0
        # 2. 양의 정부호 보장 (Jittering)
        self.ukf.P += np.eye(self.ukf.P.shape[0]) * 1e-9

        # 3. 예측 수행
        try:
            self.ukf.predict(dt=dt)
        except np.linalg.LinAlgError:
            print("UKF Predict Failed. Resetting Covariance.")  # 로거가 없으므로 print 사용
            self.ukf.P = np.eye(5) * 0.1
            self.ukf.predict(dt=dt)

    def update_icp(self, z, motion_factor=1.0):
        # 움직임이 클수록 ICP 노이즈 공분산을 키움 (Adaptive R)
        self.R_icp = self.R_icp_base * motion_factor
        self.ukf.update(z, R=self.R_icp, hx=self.hx)

    def update_imu(self, omega):
        self.ukf.update(np.array([omega]), R=self.R_imu, hx=self.hx_imu)

    def get_state(self):
        return self.ukf.x.copy()

    def get_covariance(self):
        return self.ukf.P.copy()


# --- Node Class ---
class OdomLocalizerNode(Node):
    def __init__(self):
        super().__init__('odom_localizer')

        self.scan_cb_group = MutuallyExclusiveCallbackGroup()
        self.imu_cb_group = MutuallyExclusiveCallbackGroup()
        self.lock = Lock()

        # Parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('keyframe_dist', 0.15)  # Slightly stricter
        self.declare_parameter('keyframe_angle', 0.1)  # Slightly stricter

        # Noise Params
        self.declare_parameter('ukf_process_noise', [0.001, 0.001, 0.001, 0.1, 0.1])
        self.declare_parameter('ukf_meas_noise_icp', [0.01, 0.01, 0.005])  # Trust ICP more if good
        self.declare_parameter('ukf_meas_noise_imu', [0.01])

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.kf_dist_th = self.get_parameter('keyframe_dist').value
        self.kf_angle_th = self.get_parameter('keyframe_angle').value

        self.ukf = RobotUKF(
            dt=0.01,
            Q_params=self.get_parameter('ukf_process_noise').value,
            R_icp_params=self.get_parameter('ukf_meas_noise_icp').value,
            R_imu_params=self.get_parameter('ukf_meas_noise_imu').value
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos, callback_group=self.scan_cb_group)
        self.imu_sub = self.create_subscription(
            Imu, '/imu_plugin/out', self.imu_callback, qos, callback_group=self.imu_cb_group)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Buffers
        self.imu_times = deque(maxlen=3000)
        self.imu_data = deque(maxlen=3000)

        self.keyframe_pcd = None
        self.last_keyframe_time = None
        self.last_imu_time = None
        self.last_keyframe_pose = np.zeros(3)  # [x, y, theta] Global

        # Deskewing Pre-calc
        self.cos_cache = {}
        self.sin_cache = {}

        self.get_logger().info('Improved Odom Localizer Initialized')

    def get_interpolated_omega(self, query_time):
        """Thread-safe linear interpolation of IMU yaw rate."""
        with self.lock:
            if not self.imu_times: return 0.0

            # Boundary check
            if query_time <= self.imu_times[0]: return self.imu_data[0]
            if query_time >= self.imu_times[-1]: return self.imu_data[-1]

            idx = bisect_left(self.imu_times, query_time)
            t1 = self.imu_times[idx - 1]
            t2 = self.imu_times[idx]
            w1 = self.imu_data[idx - 1]
            w2 = self.imu_data[idx]

            if t2 - t1 < 1e-9: return w1

            ratio = (query_time - t1) / (t2 - t1)
            return w1 + ratio * (w2 - w1)

    def integrate_imu_yaw(self, t_start, t_end):
        """Calculate relative rotation between two timestamps using IMU data."""
        with self.lock:
            if not self.imu_times or t_start >= t_end:
                return 0.0

            idx_start = bisect_left(self.imu_times, t_start)
            idx_end = bisect_left(self.imu_times, t_end)

            # 데이터가 충분하지 않으면 0 반환 (혹은 외삽)
            if idx_start >= len(self.imu_times): return 0.0

            # Slice safe copy
            ts = list(self.imu_times)[idx_start:idx_end + 1]
            ws = list(self.imu_data)[idx_start:idx_end + 1]

            if len(ts) < 2: return 0.0

            ts_arr = np.array(ts)
            ws_arr = np.array(ws)
            dt_arr = np.diff(ts_arr)
            avg_w = (ws_arr[:-1] + ws_arr[1:]) / 2.0

            return np.sum(avg_w * dt_arr)

    def deskew_scan(self, scan_msg, angular_velocity):
        """Vectorized Lidar Deskewing."""
        ranges = np.array(scan_msg.ranges)
        # Filter invalid ranges
        valid_mask = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)

        if np.sum(valid_mask) < 30: return None

        ranges = ranges[valid_mask]

        # Angles generation
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges), endpoint=False)[valid_mask]

        # Time offsets for each ray
        time_inc = scan_msg.time_increment
        if time_inc < 1e-9:
            time_inc = scan_msg.scan_time / len(scan_msg.ranges)

        indices = np.where(valid_mask)[0]
        dt_offsets = indices * time_inc  # Shape: (N,)

        # Deskewing: Rotate each point backwards/forwards based on angular velocity
        # 보정된 각도 = 측정 각도 + (각속도 * 시간차)
        corrected_angles = angles + (angular_velocity * dt_offsets)

        x = ranges * np.cos(corrected_angles)
        y = ranges * np.sin(corrected_angles)

        # Stack to (N, 2)
        return np.stack([x, y], axis=1)

    def scan_callback(self, msg):
        scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 1. IMU Interpolation for Deskewing
        current_omega = self.get_interpolated_omega(scan_time)
        current_pcd = self.deskew_scan(msg, current_omega)

        if current_pcd is None:
            return

        # 2. Keyframe Initialization
        if self.keyframe_pcd is None:
            self.keyframe_pcd = current_pcd
            self.last_keyframe_time = scan_time
            # 초기 위치 설정 (필요 시)
            self.last_keyframe_pose = self.ukf.get_state()[:3]
            return

        # 3. IMU Pre-integration (Guess Calculation)
        # Keyframe 시점(t_k)에서 Current Scan 시점(t_c)까지의 회전량 계산
        imu_delta_yaw = self.integrate_imu_yaw(self.last_keyframe_time, scan_time)

        # Initial Guess Matrix (Rotation only)
        c, s = np.cos(imu_delta_yaw), np.sin(imu_delta_yaw)
        R_guess = np.array([[c, -s], [s, c]])  # (2, 2)

        # 4. ICP Correction
        # 포인트 클라우드를 미리 회전시키는 대신, ICP 함수에 initial transformation을 줄 수 있다면 그게 더 좋음
        # 여기서는 기존 로직대로 '선 회전(Pre-rotation)' 후 '잔여 오차(Residual)' 계산 방식을 씀

        # (N, 2) -> (2, N) -> Rotate -> (N, 2)
        current_pcd_rotated = (R_guess @ current_pcd.T).T

        try:
            # Source: 회전된 현재 클라우드, Target: 키프레임
            # 주의: ICP는 Source를 Target에 맞추는 변환 행렬을 찾음
            correction = icp_2d(
                previous_pcd=self.keyframe_pcd,
                current_pcd=current_pcd_rotated,
                max_iterations=30,
                tolerance=1e-5,
                distance_threshold=0.3
            )
        except Exception as e:
            self.get_logger().warn(f"ICP Error: {e}")
            return

        # Extract ICP Residual Transform
        dx_res = correction[0, 2]
        dy_res = correction[1, 2]
        dtheta_res = np.arctan2(correction[1, 0], correction[0, 0])

        # 5. Transform Composition (Global Frame)
        # Total Rotation = IMU_Guess + ICP_Residual
        dtheta_total = normalize_angle(imu_delta_yaw + dtheta_res)

        # Total Translation
        # 우리가 구한 dx_res, dy_res는 "Rotated Source"를 "Keyframe"에 맞추는 값임.
        # Keyframe 기준 좌표계에서의 이동량이므로 바로 사용 가능.
        dx_total = dx_res
        dy_total = dy_res

        # 6. Outlier Rejection (Simple Gate)
        # 튀는 값 방어: 갑자기 0.5m 이상 점프하거나 30도 이상 회전 (Keyframe 간격 고려 시 큰 값)
        #if (dx_total ** 2 + dy_total ** 2) > 0.5 or abs(dtheta_total) > 0.6:
        #    self.get_logger().warn(f"Outlier rejected: dx={dx_total:.2f}, th={dtheta_total:.2f}")
        #    return

        # 7. Global Pose Calculation
        kf_x, kf_y, kf_th = self.last_keyframe_pose
        c_k, s_k = np.cos(kf_th), np.sin(kf_th)

        # Keyframe 좌표계의 (dx, dy)를 Global 좌표계로 변환하여 더함
        meas_x = kf_x + (c_k * dx_total - s_k * dy_total)
        meas_y = kf_y + (s_k * dx_total + c_k * dy_total)
        meas_theta = normalize_angle(kf_th + dtheta_total)

        measurement = np.array([meas_x, meas_y, meas_theta])

        # 8. UKF Update
        with self.lock:
            # Adaptive Noise: 회전 속도가 빠르면 ICP 결과를 덜 신뢰 (노이즈 증가)
            motion_factor = 1.0 + 5.0 * abs(current_omega)
            self.ukf.update_icp(measurement, motion_factor=motion_factor)

            current_state = self.ukf.get_state()

            # 9. Keyframe Update Logic
            dist_sq = dx_total ** 2 + dy_total ** 2

            # Keyframe 갱신 조건 만족 시
            if dist_sq > self.kf_dist_th ** 2 or abs(dtheta_total) > self.kf_angle_th:
                self.keyframe_pcd = current_pcd
                self.last_keyframe_time = scan_time

                # [중요] Keyframe의 기준 Pose를 '순수 ICP 누적'이 아닌 'UKF Fused Pose'로 재설정
                # 이를 통해 IMU가 보정한 위치로 Keyframe Anchor를 옮겨 Drift를 완화함.
                self.last_keyframe_pose = current_state[:3]

                self.get_logger().debug(f"New Keyframe: {self.last_keyframe_pose}")

    def imu_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        omega = msg.angular_velocity.z
        with self.lock:
            self.imu_times.append(current_time)
            self.imu_data.append(omega)

            if self.last_imu_time is None:
                self.last_imu_time = current_time
                return

            dt = current_time - self.last_imu_time
            if dt <= 0: return  # Skip invalid dt

            self.last_imu_time = current_time

            # predict 함수 안으로 로직을 옮겼으므로 그냥 호출만 하면 됩니다.
            self.ukf.predict(dt)

            # 2. Update (IMU direct measurement)
            self.ukf.update_imu(omega)

            # 3. Publish TF immediately for smooth visualization
            self.publish_tf(msg.header.stamp)

    def publish_tf(self, timestamp):
        state = self.ukf.get_state()  # [x, y, theta, v, omega]

        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = state[0]
        t.transform.translation.y = state[1]
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0, 0, state[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomLocalizerNode()

    # 4 threads: 1 main, 1 scan, 1 imu, 1 extra
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
