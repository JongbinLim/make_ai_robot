#!/usr/bin/env python3

"""
ROS2 Odom Localizer Node with Robust UKF Fusion & Pre-integration
[Version 2.2 - Improved TF Stability & Error Handling]
- Added Numba Warm-up to prevent initial lag
- ICP returns fitness score to reject bad matches
- TF publishing synchronized with high-frequency IMU
- Robust checks for NaN/Inf values
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

# --- Numba Imports ---
from numba import njit, float64, int32


# ==========================================
#       Numba Optimized Core Functions
# ==========================================

@njit(cache=True, fastmath=True)
def normalize_angle_jit(angle):
    """Fast angle normalization to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True, fastmath=True)
def deskew_scan_numba(ranges, angle_min, angle_increment,
                      range_min, range_max, time_increment, angular_velocity):
    """
    Vectorized & JIT-compiled Lidar Deskewing
    """
    n = len(ranges)
    points = np.empty((n, 2), dtype=np.float64)
    valid_count = 0

    for i in range(n):
        r = ranges[i]
        if r < range_min or r > range_max or np.isnan(r) or np.isinf(r):
            continue

        theta_base = angle_min + i * angle_increment
        dt = i * time_increment
        theta_corrected = theta_base + (angular_velocity * dt)

        points[valid_count, 0] = r * np.cos(theta_corrected)
        points[valid_count, 1] = r * np.sin(theta_corrected)
        valid_count += 1

    return points[:valid_count]


@njit(cache=True, fastmath=True)
def get_nearest_neighbors(src, dst, max_dist_sq):
    """
    Find nearest neighbors using brute-force
    Returns: (distances_squared, indices)
    """
    n_src = src.shape[0]
    n_dst = dst.shape[0]

    indices = np.empty(n_src, dtype=np.int32)
    dists = np.empty(n_src, dtype=np.float64)

    for i in range(n_src):
        min_d2 = 1e9
        min_idx = -1
        p_x = src[i, 0]
        p_y = src[i, 1]

        for j in range(n_dst):
            dx = p_x - dst[j, 0]
            dy = p_y - dst[j, 1]
            d2 = dx * dx + dy * dy

            if d2 < min_d2:
                min_d2 = d2
                min_idx = j

        if min_d2 > max_dist_sq:
            indices[i] = -1  # Invalid
            dists[i] = min_d2
        else:
            indices[i] = min_idx
            dists[i] = min_d2

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

    src_centered = src.copy()
    dst_centered = dst.copy()

    src_centered[:, 0] -= src_mean_x
    src_centered[:, 1] -= src_mean_y
    dst_centered[:, 0] -= dst_mean_x
    dst_centered[:, 1] -= dst_mean_y

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t_x = dst_mean_x - (R[0, 0] * src_mean_x + R[0, 1] * src_mean_y)
    t_y = dst_mean_y - (R[1, 0] * src_mean_x + R[1, 1] * src_mean_y)

    return R, t_x, t_y


@njit(cache=True)
def icp_2d_numba(previous_pcd, current_pcd, max_iterations=30, tolerance=1e-4, distance_threshold=0.5):
    """
    Numba Optimized ICP Algorithm
    Returns:
        H: 3x3 Transformation Matrix (homogeneous)
        fitness_score: Mean Squared Error (lower is better)
    """
    H = np.eye(3, dtype=np.float64)
    src = current_pcd.copy()
    dst = previous_pcd
    dist_th_sq = distance_threshold ** 2
    prev_error = 1e9

    final_mse = 1e9

    for _ in range(max_iterations):
        dists, indices = get_nearest_neighbors(src, dst, dist_th_sq)
        valid_mask = indices != -1
        valid_cnt = np.sum(valid_mask)

        if valid_cnt < 10:
            return np.eye(3, dtype=np.float64), 9999.0

        src_valid = src[valid_mask]
        dst_valid = dst[indices[valid_mask]]

        # Compute Transform
        R, tx, ty = compute_transform_svd(src_valid, dst_valid)

        # Apply to src
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

        # Check Convergence
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
    def __init__(self, dt=0.01, Q_params=None, R_icp_params=None, R_imu_params=None):
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0)
        self.ukf = UnscentedKalmanFilter(dim_x=5, dim_z=3, dt=dt, fx=self.fx, hx=self.hx, points=points)

        self.ukf.x = np.zeros(5)
        self.ukf.P = np.eye(5) * 0.1

        q_diag = Q_params if Q_params else [0.001, 0.001, 0.001, 0.05, 0.05]
        self.ukf.Q = np.diag(q_diag)

        r_icp_diag = R_icp_params if R_icp_params else [0.05, 0.05, 0.02]
        self.R_icp_base = np.diag(r_icp_diag)
        self.R_icp = self.R_icp_base.copy()

        r_imu_diag = R_imu_params if R_imu_params else [0.01]
        self.R_imu = np.diag(r_imu_diag)

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

    def fx(self, x, dt):
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

        next_theta = normalize_angle_jit(theta + omega * dt)
        # Decay helps stability when no input
        next_v = v * 0.98
        next_omega = omega * 0.95
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

    def predict(self, dt):
        # Enforce symmetry
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0
        try:
            self.ukf.predict(dt=dt)
        except Exception:
            self.ukf.P = np.eye(5) * 0.5
            self.ukf.predict(dt=dt)

    def update_icp(self, z, motion_factor=1.0):
        self.R_icp = self.R_icp_base * motion_factor
        self.ukf.update(z, R=self.R_icp, hx=self.hx)

    def update_imu(self, omega):
        self.ukf.update(np.array([omega]), R=self.R_imu, hx=self.hx_imu)

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
        self.declare_parameter('keyframe_dist', 0.3)
        self.declare_parameter('keyframe_angle', 0.3)

        # Reduced noise for better stability
        self.declare_parameter('ukf_process_noise', [0.001, 0.001, 0.001, 0.01, 0.01])
        self.declare_parameter('ukf_meas_noise_icp', [0.05, 0.05, 0.02])
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

        self.imu_times = deque(maxlen=3000)
        self.imu_data = deque(maxlen=3000)

        self.keyframe_pcd = None
        self.last_keyframe_time = None
        self.last_imu_time = None
        self.last_keyframe_pose = np.zeros(3)

        # Warm up Numba to prevent lag on first callback
        self._warmup_numba()

        self.get_logger().info('Odom Localizer Ready (Numba Warm-up Complete)')

    def _warmup_numba(self):
        """Run JIT functions with dummy data to trigger compilation"""
        self.get_logger().info('Warming up Numba kernels...')
        dummy_scan = np.random.rand(50, 2).astype(np.float64)
        dummy_ranges = np.ones(50, dtype=np.float64)

        # Warmup Deskew
        deskew_scan_numba(dummy_ranges, -1.0, 0.01, 0.1, 10.0, 0.0, 0.1)

        # Warmup ICP
        icp_2d_numba(dummy_scan, dummy_scan)
        self.get_logger().info('Warm-up Done.')

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

            # Safe slice handling
            idx_start = max(0, idx_start)
            idx_end = min(len(self.imu_times) - 1, idx_end)

            if idx_start >= idx_end: return 0.0

            ts = list(self.imu_times)
            ws = list(self.imu_data)

            sub_ts = np.array(ts[idx_start:idx_end + 1])
            sub_ws = np.array(ws[idx_start:idx_end + 1])

            if len(sub_ts) < 2: return 0.0

            dt_arr = np.diff(sub_ts)
            avg_w = (sub_ws[:-1] + sub_ws[1:]) / 2.0
            return np.sum(avg_w * dt_arr)

    def deskew_scan(self, scan_msg, angular_velocity):
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
            float(angular_velocity)
        )

        if pcd.shape[0] < 30: return None
        return pcd

    def scan_callback(self, msg):
        """
        Perform ICP Correction.
        NOTE: Do NOT publish TF here. TF is published in IMU callback for smoothness.
        """
        scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Get IMU data for deskewing
        current_omega = self.get_interpolated_omega(scan_time)
        current_pcd = self.deskew_scan(msg, current_omega)

        if current_pcd is None:
            return

        # Initialize Keyframe if needed
        if self.keyframe_pcd is None:
            self.keyframe_pcd = current_pcd
            self.last_keyframe_time = scan_time
            # Init UKF pose to match parameter if needed, but usually starts at 0
            self.last_keyframe_pose = self.ukf.get_state()[:3]
            return

        # Pre-integration for initial guess
        imu_delta_yaw = self.integrate_imu_yaw(self.last_keyframe_time, scan_time)
        c, s = np.cos(imu_delta_yaw), np.sin(imu_delta_yaw)
        R_guess = np.array([[c, -s], [s, c]])

        current_pcd_rotated = (R_guess @ current_pcd.T).T

        # --- Numba Accelerated ICP ---
        try:
            H_icp, fitness_score = icp_2d_numba(
                previous_pcd=self.keyframe_pcd,
                current_pcd=current_pcd_rotated,
                max_iterations=40,
                tolerance=1e-5,
                distance_threshold=0.5
            )
        except Exception as e:
            self.get_logger().warn(f"ICP Error: {e}")
            return

        # Simple Gate: Reject bad matches (e.g., dynamic obstacles, featureless corridors)
        if fitness_score > 0.1:  # Threshold depends on map scale/resolution
            self.get_logger().debug(f"ICP Rejected: High MSE {fitness_score:.4f}")
            return

        dx_res = H_icp[0, 2]
        dy_res = H_icp[1, 2]
        dtheta_res = np.arctan2(H_icp[1, 0], H_icp[0, 0])

        dtheta_total = normalize_angle_jit(imu_delta_yaw + dtheta_res)
        dx_total = dx_res
        dy_total = dy_res

        # Calculate Global Measurement based on Keyframe
        kf_x, kf_y, kf_th = self.last_keyframe_pose
        c_k, s_k = np.cos(kf_th), np.sin(kf_th)

        meas_x = kf_x + (c_k * dx_total - s_k * dy_total)
        meas_y = kf_y + (s_k * dx_total + c_k * dy_total)
        meas_theta = normalize_angle_jit(kf_th + dtheta_total)

        measurement = np.array([meas_x, meas_y, meas_theta])

        # Update UKF
        with self.lock:
            # Adaptive noise based on angular velocity (turning is harder)
            motion_factor = 1.0 + 3.0 * abs(current_omega)
            self.ukf.update_icp(measurement, motion_factor=motion_factor)

            current_state = self.ukf.get_state()

            # Keyframe Update Decision
            dist_sq = dx_total ** 2 + dy_total ** 2
            if dist_sq > self.kf_dist_th ** 2 or abs(dtheta_total) > self.kf_angle_th:
                self.keyframe_pcd = current_pcd
                self.last_keyframe_time = scan_time
                self.last_keyframe_pose = current_state[:3]

    def imu_callback(self, msg):
        """
        High-rate Prediction & TF Publishing
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        omega = msg.angular_velocity.z

        with self.lock:
            self.imu_times.append(current_time)
            self.imu_data.append(omega)

            if self.last_imu_time is None:
                self.last_imu_time = current_time
                return

            dt = current_time - self.last_imu_time
            if dt <= 0: return

            # Predict UKF
            # If DT is too large (lag), break it down
            if dt > 0.1:
                step = 0.05
                remain = dt
                while remain > 0:
                    d = min(step, remain)
                    self.ukf.predict(d)
                    remain -= d
            else:
                self.ukf.predict(dt)

            # Update IMU (treat as measurement for omega state)
            self.ukf.update_imu(omega)

            # Publish TF immediately with current IMU timestamp
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

        # Explicit Normalization of Quaternion
        q = quaternion_from_euler(0, 0, state[2])
        norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
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

    # Use MultiThreadedExecutor to handle Scan(ICP) and IMU(High-rate) concurrently
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
