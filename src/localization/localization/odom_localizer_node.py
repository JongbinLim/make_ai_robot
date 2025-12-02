#!/usr/bin/env python3

"""
ROS2 Odom Localizer Node with Robust UKF Fusion & Pre-integration
[Version 2.3 - Fixed Grid Indexing, Cholesky Stability, and Numba Safety]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan, Imu
from tf2_ros import TransformBroadcaster
import tf_transformations
import numpy as np
from collections import deque
from bisect import bisect_left
from threading import Lock
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from geometry_msgs.msg import TransformStamped, Twist

# --- Numba Imports ---
from numba import njit, prange


# ==========================================
#       Numba Optimized Core Functions
# ==========================================

@njit(cache=True, fastmath=True)
def normalize_angle_jit(angle):
    """Fast angle normalization to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True, fastmath=True)
def compute_distribution_numba(points):
    """
    Compute PCA of the point cloud to determine direction and spread.
    """
    n = len(points)
    if n < 10:
        return False, np.zeros(2), np.eye(2)

    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += points[i, 0]
        mean_y += points[i, 1]
    mean_x /= n
    mean_y /= n

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yy = 0.0
    for i in range(n):
        dx = points[i, 0] - mean_x
        dy = points[i, 1] - mean_y
        cov_xx += dx * dx
        cov_xy += dx * dy
        cov_yy += dy * dy

    cov_xx /= n
    cov_xy /= n
    cov_yy /= n

    cov_matrix = np.array([[cov_xx, cov_xy],
                           [cov_xy, cov_yy]])

    # w[0] = min, w[1] = max
    w, v = np.linalg.eigh(cov_matrix)
    return True, w, v


@njit(cache=True, fastmath=True)
def deskew_scan_numba(ranges, angle_min, angle_increment,
                      range_min, range_max, time_increment, angular_velocity, linear_velocity):
    """
    Vectorized & JIT-compiled Lidar Deskewing with strict NaN/Inf checks
    """
    n = len(ranges)
    points = np.empty((n, 2), dtype=np.float64)
    valid_count = 0

    # Pre-calculate constants
    inv_omega = 0.0
    if abs(angular_velocity) > 1e-4:
        inv_omega = 1.0 / angular_velocity

    for i in range(n):
        r = ranges[i]
        # Strict range and validity check
        if r < range_min or r > range_max or np.isnan(r) or np.isinf(r):
            continue

        dt = i * time_increment
        delta_theta = angular_velocity * dt

        robot_dx = 0.0
        robot_dy = 0.0

        if abs(angular_velocity) < 1e-4:
            robot_dx = linear_velocity * dt
        else:
            # Arc motion model
            radius = linear_velocity * inv_omega
            robot_dx = radius * np.sin(delta_theta)
            robot_dy = radius * (1 - np.cos(delta_theta))

        theta_base = angle_min + i * angle_increment
        theta_corrected = theta_base + delta_theta

        points[valid_count, 0] = r * np.cos(theta_corrected) + robot_dx
        points[valid_count, 1] = r * np.sin(theta_corrected) + robot_dy
        valid_count += 1

    return points[:valid_count]


@njit(cache=True, fastmath=True)
def build_grid_map(points, resolution=0.025, pad=2.0):
    """
    Build a 2D lookup grid.
    [FIX] Added boundary clamping to prevent IndexError when point is exactly at max_x.
    """
    if len(points) == 0:
        return np.full((1, 1), -1, dtype=np.int32), np.zeros(2, dtype=np.float64), 1, 1

    min_x = np.min(points[:, 0]) - pad
    min_y = np.min(points[:, 1]) - pad
    max_x = np.max(points[:, 0]) + pad
    max_y = np.max(points[:, 1]) + pad

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Safety clamp for width/height to avoid zero size
    width = max(width, 1)
    height = max(height, 1)

    grid = np.full((width, height), -1, dtype=np.int32)
    inv_res = 1.0 / resolution

    for i in range(len(points)):
        gx = int((points[i, 0] - min_x) * inv_res)
        gy = int((points[i, 1] - min_y) * inv_res)

        # [CRITICAL FIX] Clamp indices to valid range
        if gx >= width: gx = width - 1
        if gy >= height: gy = height - 1
        if gx < 0: gx = 0
        if gy < 0: gy = 0

        grid[gx, gy] = i

    min_xy = np.array([min_x, min_y], dtype=np.float64)
    return grid, min_xy, width, height


@njit(cache=True, fastmath=True, parallel=True)
def get_correspondences_grid(src, dst, grid, min_xy, grid_shape, resolution, max_dist_sq):
    n_src = src.shape[0]
    indices = np.full(n_src, -1, dtype=np.int32)
    dists = np.zeros(n_src, dtype=np.float64)

    grid_w, grid_h = grid_shape
    min_x, min_y = min_xy[0], min_xy[1]
    inv_res = 1.0 / resolution

    search_radius = 4  # 0.05m * 4 = 20cm 범위 탐색

    for i in prange(n_src):
        px = src[i, 0]
        py = src[i, 1]

        gx = int((px - min_x) * inv_res)
        gy = int((py - min_y) * inv_res)

        best_dist = max_dist_sq
        best_idx = -1

        # Check neighbor grids
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = gx + dx, gy + dy

                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    candidate_idx = grid[nx, ny]
                    if candidate_idx != -1:
                        dx_real = px - dst[candidate_idx, 0]
                        dy_real = py - dst[candidate_idx, 1]
                        d2 = dx_real * dx_real + dy_real * dy_real

                        if d2 < best_dist:
                            best_dist = d2
                            best_idx = candidate_idx

        dists[i] = best_dist
        indices[i] = best_idx

    return dists, indices


@njit(cache=True)
def compute_transform_svd(src, dst):
    """Compute rigid 2D transform (R, t) using SVD"""
    n = src.shape[0]
    if n == 0:
        return np.eye(2), 0.0, 0.0

    src_mean_x = np.sum(src[:, 0]) / n
    src_mean_y = np.sum(src[:, 1]) / n
    dst_mean_x = np.sum(dst[:, 0]) / n
    dst_mean_y = np.sum(dst[:, 1]) / n

    # Centering (Avoid making full copies if possible, but Numba optimizes this)
    src_c = src - np.array([src_mean_x, src_mean_y])
    dst_c = dst - np.array([dst_mean_x, dst_mean_y])

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t_x = dst_mean_x - (R[0, 0] * src_mean_x + R[0, 1] * src_mean_y)
    t_y = dst_mean_y - (R[1, 0] * src_mean_x + R[1, 1] * src_mean_y)

    return R, t_x, t_y


@njit(cache=True, fastmath=True, nogil=True)
def icp_2d_grid_numba(previous_pcd, current_pcd, grid, min_xy, grid_shape, max_iterations=30, tolerance=1e-3, distance_threshold=0.5):
    """
    Optimized ICP using Grid Look-up.
    """
    H = np.eye(3, dtype=np.float64)
    src = current_pcd.copy()
    dst = previous_pcd

    dist_th_sq = distance_threshold ** 2
    prev_error = 1e9
    final_mse = 1e9

    for _ in range(max_iterations):
        dists, indices = get_correspondences_grid(
            src, dst, grid, min_xy, grid_shape, 0.05, dist_th_sq
        )

        valid_mask = indices != -1
        valid_cnt = np.sum(valid_mask)

        if valid_cnt < 10:
            return np.eye(3, dtype=np.float64), 9999.0

        # [CRITICAL] Boolean indexing in Numba works, ensuring only valid points are used
        src_valid = src[valid_mask]
        dst_valid = dst[indices[valid_mask]]

        R, tx, ty = compute_transform_svd(src_valid, dst_valid)

        # Apply to src (In-place update for next iteration)
        src_new_x = R[0, 0] * src[:, 0] + R[0, 1] * src[:, 1] + tx
        src_new_y = R[1, 0] * src[:, 0] + R[1, 1] * src[:, 1] + ty
        src[:, 0] = src_new_x
        src[:, 1] = src_new_y

        # Update Accumulator
        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[0, 2] = tx
        T_step[1, 2] = ty
        H = T_step @ H

        current_mse = np.sum(dists[valid_mask]) / valid_cnt
        final_mse = current_mse

        if np.abs(prev_error - current_mse) < tolerance:
            break
        prev_error = current_mse

    return H, final_mse


# ==========================================
#       End of Numba Functions
# ==========================================

def quaternion_from_euler(roll, pitch, yaw):
    return tf_transformations.quaternion_from_euler(roll, pitch, yaw)


class RobotUKF:
    def __init__(self, dt=0.05, Q_params=None, R_icp_params=None, R_imu_params=None):
        # [TUNING] kappa=0 for dim=5 is generally safe.
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0)
        self.ukf = UnscentedKalmanFilter(dim_x=5, dim_z=3, dt=dt, fx=self.fx, hx=self.hx, points=points)

        self.ukf.x = np.zeros(5)
        self.ukf.P = np.eye(5) * 0.1

        self.dt_default = dt
        q_diag = Q_params if Q_params else [0.001, 0.001, 0.001, 0.01, 0.05]
        self.Q_base = np.diag(q_diag)
        self.ukf.Q = self.Q_base.copy()

        r_icp_diag = R_icp_params if R_icp_params else [0.05, 0.05, 0.02]
        self.R_icp_base = np.diag(r_icp_diag)

        r_imu_diag = R_imu_params if R_imu_params else [0.02]
        self.R_imu = np.diag(r_imu_diag)

        # Assign mean functions
        self.ukf.x_mean = self.state_mean
        self.ukf.z_mean = self.measurement_mean
        self.ukf.residual_x = self.residual_x
        self.ukf.residual_z = self.residual_h

    def state_mean(self, sigmas, Wm):
        x = np.zeros(5)
        x[0] = np.dot(sigmas[:, 0], Wm)
        x[1] = np.dot(sigmas[:, 1], Wm)
        x[3] = np.dot(sigmas[:, 3], Wm)
        x[4] = np.dot(sigmas[:, 4], Wm)

        sin_sum = np.dot(np.sin(sigmas[:, 2]), Wm)
        cos_sum = np.dot(np.cos(sigmas[:, 2]), Wm)
        x[2] = np.arctan2(sin_sum, cos_sum)
        return x

    def measurement_mean(self, sigmas, Wm):
        dim_z = sigmas.shape[1]
        z = np.zeros(dim_z)
        if dim_z == 3:  # ICP
            z[0] = np.dot(sigmas[:, 0], Wm)
            z[1] = np.dot(sigmas[:, 1], Wm)
            sin_sum = np.dot(np.sin(sigmas[:, 2]), Wm)
            cos_sum = np.dot(np.cos(sigmas[:, 2]), Wm)
            z[2] = np.arctan2(sin_sum, cos_sum)
        else:  # IMU
            z[0] = np.dot(sigmas[:, 0], Wm)
        return z

    def fx(self, x, dt, u):
        theta, v, omega = x[2], x[3], x[4]
        cmd_v, cmd_omega = u[0], u[1]

        # Control input mixing
        alpha_v = 0.1 # 튜닝!
        alpha_w = 0.0 # 튜닝!
        next_v = v + alpha_v * (cmd_v - v)
        next_omega = omega + alpha_w * (cmd_omega - omega)

        if abs(omega) > 1e-5:
            s_t = np.sin(theta)
            c_t = np.cos(theta)
            s_t_next = np.sin(theta + omega * dt)
            c_t_next = np.cos(theta + omega * dt)
            next_x = x[0] + (v / omega) * (s_t_next - s_t)
            next_y = x[1] + (v / omega) * (-c_t_next + c_t)
        else:
            # 직진 주행 시 runge-kutta 2nd order 근사
            next_x = x[0] + v * np.cos(theta + omega * dt * 0.5) * dt
            next_y = x[1] + v * np.sin(theta + omega * dt * 0.5) * dt

        next_theta = normalize_angle_jit(theta + omega * dt)

        return np.array([next_x, next_y, next_theta, next_v, next_omega])

    def hx(self, x):
        return np.array([x[0], x[1], x[2]])

    def hx_imu(self, x):
        return np.array([x[4]])

    def residual_x(self, a, b):
        y = a - b
        y[2] = normalize_angle_jit(y[2])
        return y

    def residual_h(self, a, b):
        y = a - b
        if len(y) == 3:
            y[2] = normalize_angle_jit(y[2])
        return y

    def predict(self, dt, u=np.zeros(2)):
        # Force Symmetry to prevent non-positive definite errors
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0
        # Add small noise to diagonal to prevent non-positive definite
        self.ukf.P += np.eye(5) * 1e-6

        if dt > 1e-6:
            scale_factor = dt / self.dt_default
            self.ukf.Q = self.Q_base * scale_factor
        else:
            self.ukf.Q = self.Q_base

        try:
            # Pass 'u' explicitly as kwarg, which filterpy passes to fx(x, dt, **kwargs)
            self.ukf.predict(dt=dt, u=u)
        except Exception as e:
            print(f"[UKF Predict Error] {e} - Resetting P")
            self.ukf.P = np.eye(5) * 0.1
            self.ukf.predict(dt=dt, u=u)

    def update_icp(self, z, eigenvalues, eigenvectors, motion_factor=1.0):
        # Stabilize before update
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0

        lambda_min = max(eigenvalues[0], 1e-6)
        lambda_max = eigenvalues[1]
        ratio = lambda_max / lambda_min

        base_noise_x = self.R_icp_base[0, 0] * motion_factor
        long_axis_scale = 1.0 + np.log1p(max(0, ratio - 3.0)) * 10.0
        long_axis_scale = min(long_axis_scale, 100.0)

        sigma_short = base_noise_x
        sigma_long = base_noise_x * long_axis_scale

        # Covariance in Principal Component Frame
        R_pc_aligned = np.array([[sigma_short, 0.0],
                                 [0.0, sigma_long]])

        # Transform to Local Frame (Body)
        R_local = eigenvectors @ R_pc_aligned @ eigenvectors.T

        # Transform to Global Frame
        current_yaw = self.ukf.x[2]
        c, s = np.cos(current_yaw), np.sin(current_yaw)
        R_rot = np.array([[c, -s],
                          [s, c]])

        R_global_2d = R_rot @ R_local @ R_rot.T

        R_final = np.eye(3)
        R_final[:2, :2] = R_global_2d
        R_final[2, 2] = self.R_icp_base[2, 2] * motion_factor

        # Ensure R is symmetric and positive definite
        R_final = (R_final + R_final.T) / 2.0
        R_final += np.eye(3) * 1e-6  # 측정 노이즈에도 아주 작은 값 추가

        try:
            self.ukf.update(z, R=R_final, hx=self.hx)
        except np.linalg.LinAlgError:
            print("[UKF Update Error] LinAlgError during ICP update. Skipping.")
            return long_axis_scale  # Skip update but return scale for debug

        return long_axis_scale

    def update_imu(self, omega):
        try:
            self.ukf.update(np.array([omega]), R=self.R_imu, hx=self.hx_imu)
        except np.linalg.LinAlgError:
            pass

    def get_state(self):
        return self.ukf.x.copy()


class OdomLocalizerNode(Node):
    def __init__(self):
        super().__init__('odom_localizer')

        self.scan_cb_group = MutuallyExclusiveCallbackGroup()
        self.imu_cb_group = MutuallyExclusiveCallbackGroup()
        self.lock = Lock()

        # Parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('keyframe_dist', 0.1)
        self.declare_parameter('keyframe_angle', 0.1)
        self.declare_parameter('ukf_process_noise', [0.001, 0.001, 0.001, 0.01, 0.05])
        self.declare_parameter('ukf_meas_noise_icp', [0.05, 0.05, 0.02])
        self.declare_parameter('ukf_meas_noise_imu', [0.02])

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.kf_dist_th = self.get_parameter('keyframe_dist').value
        self.kf_angle_th = self.get_parameter('keyframe_angle').value

        self.ukf = RobotUKF(
            dt=0.05,
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
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.current_cmd = np.zeros(2)
        # Fix: Init with current time to avoid large dt on startup
        self.last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

        self.tf_broadcaster = TransformBroadcaster(self)

        self.imu_times = deque(maxlen=3000)
        self.imu_data = deque(maxlen=3000)

        self.keyframe_pcd = None
        # Grid Map 캐싱용 변수
        self.kf_grid = None
        self.kf_min_xy = None
        self.kf_grid_shape = None

        self.last_keyframe_time = None
        self.last_imu_time = None
        self.last_keyframe_pose = np.zeros(3)

        self._warmup_numba()
        self.get_logger().info('Odom Localizer Ready (Numba Warm-up Complete)')

    def _warmup_numba(self):
        """Run JIT functions with dummy data to trigger compilation"""
        self.get_logger().info('Warming up Numba kernels...')
        dummy_scan = np.random.rand(50, 2).astype(np.float64)
        dummy_ranges = np.ones(50, dtype=np.float64)
        deskew_scan_numba(dummy_ranges, -1.0, 0.01, 0.1, 10.0, 0.0, 0.1, 0.1)
        dummy_grid, dummy_min_xy, w, h = build_grid_map(dummy_scan, resolution=0.025, pad=2.0)
        dummy_grid_shape = (w, h)
        icp_2d_grid_numba(dummy_scan, dummy_scan, dummy_grid, dummy_min_xy, dummy_grid_shape)
        compute_distribution_numba(dummy_scan)
        self.get_logger().info('Warm-up Done.')

    def get_time_sec(self, stamp_msg):
        """ROS Time Msg를 float seconds로 안전하게 변환"""
        # stamp_msg가 Time 객체인 경우와 msg.header.stamp인 경우 모두 처리
        if hasattr(stamp_msg, 'sec'):
            return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        else:
            # rclpy.time.Time 객체인 경우
            return stamp_msg.nanoseconds * 1e-9

    def get_current_control(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_cmd_time > 0.5:
            return np.zeros(2)
        return self.current_cmd

    def get_interpolated_omega(self, query_time):
        with self.lock:
            if not self.imu_times: return 0.0
            if query_time <= self.imu_times[0]: return self.imu_data[0]
            if query_time >= self.imu_times[-1]: return self.imu_data[-1]

            idx = bisect_left(self.imu_times, query_time)
            t1, t2 = self.imu_times[idx - 1], self.imu_times[idx]
            w1, w2 = self.imu_data[idx - 1], self.imu_data[idx]

            if t2 - t1 < 1e-9: return w1
            ratio = (query_time - t1) / (t2 - t1)
            return w1 + ratio * (w2 - w1)

    def integrate_imu_yaw(self, t_start, t_end):
        with self.lock:
            if not self.imu_times or t_start >= t_end:
                return 0.0
            if t_end <= self.imu_times[0] or t_start >= self.imu_times[-1]:
                return 0.0

            idx_start = bisect_left(self.imu_times, t_start)
            idx_end = bisect_left(self.imu_times, t_end)

            idx_start = max(0, idx_start)
            idx_end = min(len(self.imu_times) - 1, idx_end)

            if idx_start >= idx_end: return 0.0

            # Safe conversion to list for slicing
            ts = list(self.imu_times)
            ws = list(self.imu_data)

            # Ensure slice is valid
            sub_ts = np.array(ts[idx_start:idx_end + 1])
            sub_ws = np.array(ws[idx_start:idx_end + 1])

            if len(sub_ts) < 2: return 0.0

            dt_arr = np.diff(sub_ts)
            avg_w = (sub_ws[:-1] + sub_ws[1:]) / 2.0
            return np.sum(avg_w * dt_arr)

    def deskew_scan(self, scan_msg, angular_velocity, linear_velocity):
        ranges = np.array(scan_msg.ranges, dtype=np.float64)
        time_inc = scan_msg.time_increment
        if time_inc < 1e-9:
            if len(ranges) > 0:
                time_inc = scan_msg.scan_time / len(ranges)
            else:
                time_inc = 0.0

        pcd = deskew_scan_numba(
            ranges,
            float(scan_msg.angle_min),
            float(scan_msg.angle_increment),
            float(scan_msg.range_min),
            float(scan_msg.range_max),
            float(time_inc),
            float(angular_velocity),
            float(linear_velocity)
        )

        if pcd.shape[0] < 30: return None
        return pcd

    def cmd_vel_callback(self, msg):
        self.current_cmd[0] = msg.linear.x
        self.current_cmd[1] = msg.angular.z
        self.last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def get_transform_matrix(self, x, y, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, x],
            [s, c, y],
            [0, 0, 1]
        ])

    def scan_callback(self, msg):
        scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # UKF state access needs lock
        with self.lock:
            current_state = self.ukf.get_state()
            current_v = self.ukf.get_state()[3]

        current_omega = self.get_interpolated_omega(scan_time)
        current_pcd = self.deskew_scan(msg, current_omega, current_v)

        if current_pcd is None: return

        valid, eig_vals, eig_vecs = compute_distribution_numba(current_pcd)
        if not valid: return

        if self.keyframe_pcd is None:
            self.keyframe_pcd = current_pcd
            self.last_keyframe_time = scan_time
            # Grid 미리 빌드 (여기서 딱 한 번)
            self.kf_grid, self.kf_min_xy, w, h = build_grid_map(self.keyframe_pcd, resolution=0.025, pad=2.0)
            self.kf_grid_shape = (w, h)
            with self.lock:
                self.last_keyframe_pose = self.ukf.get_state()[:3]
            return

        dt_kf = scan_time - self.last_keyframe_time
        if dt_kf <= 0: return  # Prevent backward time issues

        imu_delta_yaw = self.integrate_imu_yaw(self.last_keyframe_time, scan_time)

        pred_v = current_v

        dist_pred = pred_v * dt_kf
        dx_pred = dist_pred * np.cos(imu_delta_yaw / 2.0)  # 단순 근사
        dy_pred = dist_pred * np.sin(imu_delta_yaw / 2.0)

        # 예측된 상대 변환 행렬 (T_pred)
        # Keyframe 좌표계 기준에서 Current Frame이 어디에 있을지 예측
        T_pred = np.eye(3)
        c_i, s_i = np.cos(imu_delta_yaw), np.sin(imu_delta_yaw)
        T_pred[0, 0], T_pred[0, 1] = c_i, -s_i
        T_pred[1, 0], T_pred[1, 1] = s_i, c_i
        T_pred[0, 2] = dx_pred
        T_pred[1, 2] = dy_pred

        R_pred = T_pred[:2, :2]
        t_pred = T_pred[:2, 2]

        current_pcd_transformed = (R_pred @ current_pcd.T).T + t_pred

        try:
            H_icp, fitness_score = icp_2d_grid_numba(
                previous_pcd=self.keyframe_pcd,
                current_pcd=current_pcd_transformed,
                grid=self.kf_grid,
                min_xy=self.kf_min_xy,
                grid_shape=self.kf_grid_shape,
                max_iterations=25,
                tolerance=1e-3,
                distance_threshold=0.5
            )
        except Exception as e:
            self.get_logger().warn(f"ICP Error: {e}")
            return

        if fitness_score > 0.15:
            # ICP 매칭은 실패했더라도, 로봇이 Keyframe에서 너무 멀어졌다면(예: 10cm 이상)
            # 현재 위치(UKF 예측값)를 기준으로 Keyframe을 강제 갱신해야 함.
            # 그렇지 않으면 10m를 이동해도 계속 10m 전의 Keyframe과 매칭을 시도하다 영원히 실패함.
            with self.lock:
                current_state_pred = self.ukf.get_state()

            # 예측된 위치와 마지막 Keyframe 사이의 거리 계산
            pred_dx = current_state_pred[0] - self.last_keyframe_pose[0]
            pred_dy = current_state_pred[1] - self.last_keyframe_pose[1]
            pred_dist_sq = pred_dx ** 2 + pred_dy ** 2

            # 기준 거리(kf_dist_th)보다 많이 이동했으면 강제 갱신
            if pred_dist_sq > self.kf_dist_th ** 2:
                self.get_logger().warn(
                    f"ICP Failed (Score: {fitness_score:.3f}) but moved far. Forcing Keyframe Update.")
                self.keyframe_pcd = current_pcd
                self.last_keyframe_time = scan_time
                self.last_keyframe_pose = current_state_pred[:3]

                # Grid Map도 갱신
                self.kf_grid, self.kf_min_xy, w, h = build_grid_map(self.keyframe_pcd, resolution=0.025, pad=2.0)
                self.kf_grid_shape = (w, h)
            return

        # 3x3 행렬 곱셈으로 최종 상대 변환 계산
        T_total = H_icp @ T_pred

        dx_total = T_total[0, 2]
        dy_total = T_total[1, 2]
        dtheta_total = np.arctan2(T_total[1, 0], T_total[0, 0])

        # Global Frame에서의 측정치 계산
        kf_x, kf_y, kf_th = self.last_keyframe_pose
        T_keyframe_global = self.get_transform_matrix(kf_x, kf_y, kf_th)

        # T_current_global = T_keyframe_global * T_total
        T_meas_global = T_keyframe_global @ T_total

        meas_x = T_meas_global[0, 2]
        meas_y = T_meas_global[1, 2]
        meas_theta = np.arctan2(T_meas_global[1, 0], T_meas_global[0, 0])

        measurement = np.array([meas_x, meas_y, meas_theta])

        with self.lock:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            lag_time = now_sec - scan_time

            # lag_time이 음수(미래에서 옴)거나 1초 이상 차이나면 시계가 안 맞는 것으로 간주
            if lag_time < 0.0 or lag_time > 1.0:
                # 로깅을 한 번만 하거나 디버그로 낮추는 것이 좋음
                self.get_logger().debug(f"Time sync warning: lag={lag_time:.4f}s. Forcing lag to 0.")
                lag_time = 0.0

            # Lag가 클수록 측정치 노이즈를 키움 (Trust less if laggy)
            lag_penalty = 1.0 + (lag_time * 5.0)
            lag_penalty = min(max(lag_penalty, 1.0), 10.0)

            motion_factor = 1.0 + 3.0 * abs(current_omega) + 2.0 * abs(current_v) * lag_penalty

            # UKF Update
            self.ukf.update_icp(
                measurement,
                eigenvalues=eig_vals,
                eigenvectors=eig_vecs,
                motion_factor=motion_factor
            )

            current_state = self.ukf.get_state()

            dist_sq = dx_total ** 2 + dy_total ** 2
            if dist_sq > self.kf_dist_th ** 2 or abs(dtheta_total) > self.kf_angle_th:
                self.keyframe_pcd = current_pcd
                self.last_keyframe_time = scan_time
                self.last_keyframe_pose = current_state[:3]
                # Keyframe이 바뀌었으니 Grid도 새로 빌드
                self.kf_grid, self.kf_min_xy, w, h = build_grid_map(self.keyframe_pcd, resolution=0.025, pad=2.0)
                self.kf_grid_shape = (w, h)

    def imu_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        omega = msg.angular_velocity.z
        u = self.get_current_control()

        with self.lock:
            if self.last_imu_time is None:
                self.last_imu_time = current_time
                self.imu_times.append(current_time)
                self.imu_data.append(omega)
                return

            dt = current_time - self.last_imu_time

            # Filter bad timestamps
            if dt <= 0:
                return

            self.imu_times.append(current_time)
            self.imu_data.append(omega)

            MAX_STEP = 0.05
            remain = dt
            while remain > 1e-6:
                d = min(MAX_STEP, remain)
                self.ukf.predict(dt=d, u=u)
                remain -= d

            self.ukf.update_imu(omega)
            self.publish_tf(msg.header.stamp)
            self.last_imu_time = current_time

    def publish_tf(self, timestamp):
        state = self.ukf.get_state()
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = float(state[0])
        t.transform.translation.y = float(state[1])
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0, 0, state[2])
        # Safe normalization
        norm = np.sqrt(np.sum(np.array(q) ** 2))
        if norm > 1e-6:
            q = q / norm
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])

        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomLocalizerNode()
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
