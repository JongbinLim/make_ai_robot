import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit, prange


# ------------------------------------------------------------------------------
# Numba Optimized Core Functions
# 클래스 밖으로 빼내어 JIT 컴파일을 수행합니다.
# ------------------------------------------------------------------------------

@njit(cache=True)
def _normalize_angles_core(angles):
    """Yaw를 -pi ~ pi범위로 정규화"""
    return (angles + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True)
def _predict_core(particles, dx, dy, dyaw, motion_noise_params):
    """Motion Model Core Logic"""
    n = len(particles)

    # 랜덤 노이즈 생성 (Numba는 np.random 지원함)
    noise_x = np.random.normal(0, motion_noise_params[0], n)
    noise_y = np.random.normal(0, motion_noise_params[1], n)
    noise_yaw = np.random.normal(0, motion_noise_params[2], n)

    # 현재 파티클 방향
    p_yaw = particles[:, 2]
    c = np.cos(p_yaw)
    s = np.sin(p_yaw)

    # 로봇의 이동량에 노이즈를 더함
    noisy_dx = dx + noise_x
    noisy_dy = dy + noise_y
    noisy_dyaw = dyaw + noise_yaw

    # 로봇 좌표계(Local) -> 월드 좌표계(Global) 변환 및 이동
    particles[:, 0] += (noisy_dx * c - noisy_dy * s)
    particles[:, 1] += (noisy_dx * s + noisy_dy * c)
    particles[:, 2] += noisy_dyaw

    # 각도 정규화
    particles[:, 2] = _normalize_angles_core(particles[:, 2])

    return particles


@njit(cache=True, parallel=True, fastmath=True)
def _update_core(particles, laser_x, laser_y,
                 map_flat, map_w, map_h, map_res, map_origin_x, map_origin_y,
                 penalty_idx):
    """
    Likelihood Field Model Update Core Logic
    parallel=True를 사용하여 파티클별 계산을 병렬화합니다.
    """
    n_particles = particles.shape[0]
    n_rays = laser_x.shape[0]

    inv_res = 1.0 / map_res

    # 결과 저장용 배열 (Log scores)
    total_log_scores = np.zeros(n_particles, dtype=np.float32)

    # 모든 파티클에 대해 병렬 수행 (Vectorization 대신 Loop 사용으로 메모리 절약)
    for i in prange(n_particles):
        px = particles[i, 0]
        py = particles[i, 1]
        pth = particles[i, 2]

        c = np.cos(pth)
        s = np.sin(pth)

        sum_log_score = 0.0

        # 각 레이(Ray)에 대해 수행
        for j in range(n_rays):
            # 회전 변환 + 평행 이동 = 월드 좌표계상의 라이다 점들
            wx = px + (c * laser_x[j] - s * laser_y[j])
            wy = py + (s * laser_x[j] + c * laser_y[j])

            # 월드 좌표 -> 맵 그리드 인덱스(x, y) 변환
            mx = int((wx - map_origin_x) * inv_res)
            my = int((wy - map_origin_y) * inv_res)

            # 맵 범위 체크 및 인덱스 계산
            if 0 <= mx < map_w and 0 <= my < map_h:
                flat_idx = my * map_w + mx
            else:
                flat_idx = penalty_idx

            # Log Score 누적
            sum_log_score += map_flat[flat_idx]

        total_log_scores[i] = sum_log_score

    return total_log_scores


@njit(cache=True)
def _resample_core(particles, weights, min_particles, max_particles, kld_err, kld_z):
    """KLD-Sampling Core Logic"""

    # 현재 베스트 파티클 백업
    best_idx = np.argmax(weights)
    best_particle = particles[best_idx].copy()

    # KLD 기반 파티클 수 계산
    xy_res = 0.2
    yaw_res = np.pi / 18.0  # 약 10도

    # 각 파티클의 Bin 인덱스 계산
    k_x = np.floor(particles[:, 0] / xy_res).astype(np.int64)
    k_y = np.floor(particles[:, 1] / xy_res).astype(np.int64)
    k_yaw = np.floor(particles[:, 2] / yaw_res).astype(np.int64)

    # 3D 좌표를 1D 정수(Hash)로 압축
    # Numba에서는 array 연산이 최적화되어 있음
    bins_flat = k_x + (k_y * 100000) + (k_yaw * 10000000000)

    # 1D Unique 연산
    unique_bins = np.unique(bins_flat)
    k = len(unique_bins)

    # KLD 공식에 의한 목표 파티클 수 계산
    if k > 1:
        term1 = 1.0 - 2.0 / (9.0 * (k - 1))
        term2 = np.sqrt(2.0 / (9.0 * (k - 1))) * kld_z
        term3 = term1 + term2
        new_n_calc = int((k - 1) / (2.0 * kld_err) * (term3 ** 3))
    else:
        new_n_calc = min_particles

    new_n = max(min_particles, min(max_particles, new_n_calc))

    # Systematic Resampling
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0 + 1e-6  # 부동소수점 오차 보정

    step = 1.0 / new_n
    r = np.random.uniform(0, step)
    points = np.arange(new_n, dtype=np.float32) * step + r

    indices = np.searchsorted(cumsum, points)

    # 선택된 파티클 추출
    new_particles = particles[indices]

    # Best Particle 복구
    new_particles[0] = best_particle

    return new_particles, new_n


@njit(cache=True)
def _get_estimated_pose_core(particles, weights, best_idx):
    """Robust Average Core Logic"""
    best_particle = particles[best_idx]
    search_radius = 1.0
    search_radius_sq = search_radius ** 2

    dx = particles[:, 0] - best_particle[0]
    dy = particles[:, 1] - best_particle[1]
    dist_sq = dx ** 2 + dy ** 2

    # Mask 생성 (Boolean Indexing)
    mask = dist_sq < search_radius_sq

    # 반경 내 유효 파티클 개수 확인
    mask_count = np.sum(mask)
    if mask_count <= 1:
        return best_particle

    cluster_particles = particles[mask]
    cluster_weights = weights[mask]

    weight_sum = np.sum(cluster_weights)
    if weight_sum < 1e-15:
        return best_particle

    # 정규화
    cluster_weights = cluster_weights / weight_sum

    # 가중 평균
    x = np.sum(cluster_particles[:, 0] * cluster_weights)
    y = np.sum(cluster_particles[:, 1] * cluster_weights)

    sin_sum = np.sum(np.sin(cluster_particles[:, 2]) * cluster_weights)
    cos_sum = np.sum(np.cos(cluster_particles[:, 2]) * cluster_weights)
    yaw = np.arctan2(sin_sum, cos_sum)

    return np.array([x, y, yaw], dtype=np.float32)


# ------------------------------------------------------------------------------
# Optimized Particle Filter Class
# ------------------------------------------------------------------------------

class ParticleFilter:
    def __init__(self,
                 min_particles=200,
                 max_particles=3000,
                 initial_noise=[0.1, 0.1, 0.1]):

        self.min_particles = min_particles
        self.max_particles = max_particles
        self.num_particles = max_particles

        # 메모리 재할당 방지를 위한 고정 크기 버퍼 생성
        self.particles_buffer = np.zeros((self.max_particles, 3), dtype=np.float32)
        # 파티클 저장소 (x, y, yaw), 효율적 연산을 위해 float32를 사용
        self.particles = self.particles_buffer[:self.num_particles]
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

        # 노이즈 파라미터
        self.initial_noise = np.array(initial_noise, dtype=np.float32)
        # [x, y, yaw]에 대한 모션 노이즈 (튜닝 필요)
        self.motion_noise = np.array([0.1, 0.1, 0.05], dtype=np.float32)

        # 센서 모델 파라미터
        self.sensor_sigma = 0.1  # 가우시안 분포의 표준편차 (m)
        # Log 계산을 피하기 위해 미리 상수 계산
        self.sensor_model_factor = -0.5 / (self.sensor_sigma ** 2)

        # KLD 파라미터
        self.kld_err = 0.05  # 오차 허용 범위
        self.kld_z = 2.326  # z_0.99

        # 맵 데이터 저장 변수
        self.log_likelihood_map_flat = None
        self.map_info = None
        self.map_resolution = 0.05
        self.map_origin = np.array([0, 0], dtype=np.float32)
        self.map_width = 0
        self.map_height = 0
        self.map_size = 0

        # 최적화를 위한 룩업 테이블용 패널티 인덱스
        self.penalty_idx = 0

        # 빠른 랜덤 생성을 위한 빈 공간 캐시
        self.free_space_indices = None

        # 라이다 삼각함수 캐싱 변수
        self.cached_n_scans = -1
        self.cached_angle_min = 0.0
        self.cached_angle_inc = 0.0
        self.cached_step = 0

        # 라이다 삼각함수 캐싱 (최적화)
        self.cached_scan_angles = None
        self.full_sin_cache = None
        self.full_cos_cache = None

        # 레이다 데이터 다운 샘플링 변수. 높을수록 빨라짐
        self.scan_step = 5

    def initialize(self, x, y, yaw):
        """초기 위치(x, y, yaw) 주변에 파티클을 가우시안 분포로 뿌립니다."""
        self.num_particles = self.max_particles
        self.particles = self.particles_buffer[:self.num_particles]

        # Numba 친화적으로 변경 (직접 할당)
        self.particles[:, 0] = np.random.normal(x, self.initial_noise[0], self.num_particles)
        self.particles[:, 1] = np.random.normal(y, self.initial_noise[1], self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, self.initial_noise[2], self.num_particles)

        # Numba 함수 호출
        self.particles[:, 2] = _normalize_angles_core(self.particles[:, 2])
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def set_map(self, msg):
        """
        ROS OccupancyGrid를 받아 Likelihood Field(거리장)로 변환합니다.
        (이 함수는 초기화 시 한 번만 실행되므로 SciPy 의존성을 유지하며 JIT을 사용하지 않습니다)
        """
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if width <= 0 or height <= 0 or resolution <= 0:
            return

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        raw_data = np.array(msg.data, dtype=np.int8).reshape(height, width)
        self.map_resolution = resolution
        self.map_origin = np.array([origin_x, origin_y], dtype=np.float32)
        self.map_width = width
        self.map_height = height
        self.map_size = width * height

        # ROS맵: 0(Free), 100(Occupied), -1(Unknown)
        binary_free = (raw_data >= 0) & (raw_data < 50)

        # 빈 공간 인덱스 캐싱
        y_idxs, x_idxs = np.where(binary_free)
        self.free_space_indices = np.column_stack((x_idxs, y_idxs)).astype(np.float32)

        # EDT 계산
        dist_map_pixels = distance_transform_edt(binary_free)
        dist_map_meters = dist_map_pixels * resolution

        log_likelihood_map = (dist_map_meters ** 2) * self.sensor_model_factor

        min_log_prob = -20.0
        log_likelihood_map = np.maximum(log_likelihood_map, min_log_prob)

        # 맵 밖 참조를 위한 패딩 전략
        self.log_likelihood_map_flat = np.append(log_likelihood_map.ravel(), min_log_prob).astype(np.float32)
        self.penalty_idx = self.map_size

    def predict(self, dx, dy, dyaw):
        """Motion Model: JIT Function Call"""
        # Numba 코어 함수 호출
        self.particles = _predict_core(self.particles, dx, dy, dyaw, self.motion_noise)

    def _update_trig_cache(self, n_scans, angle_min, angle_inc, step):
        """삼각함수 테이블을 재계산합니다."""
        self.cached_n_scans = n_scans
        self.cached_angle_min = angle_min
        self.cached_angle_inc = angle_inc
        self.cached_step = step

        angles = np.arange(n_scans, dtype=np.float32)[::step] * angle_inc + angle_min

        self.full_cos_cache = np.cos(angles)
        self.full_sin_cache = np.sin(angles)

    def _recover_from_kidnapping(self):
        """모든 파티클이 길을 잃었을 때 수행합니다."""
        if self.free_space_indices is None:
            self.weights = np.ones(self.num_particles) / self.num_particles
            return

        n_recovery = self.num_particles
        num_free_cells = self.free_space_indices.shape[0]

        rand_indices = np.random.choice(num_free_cells, size=n_recovery)
        chosen_cells = self.free_space_indices[rand_indices]

        x_coords = chosen_cells[:, 0] * self.map_resolution + self.map_origin[0]
        y_coords = chosen_cells[:, 1] * self.map_resolution + self.map_origin[1]

        x_coords += np.random.uniform(0, self.map_resolution, n_recovery)
        y_coords += np.random.uniform(0, self.map_resolution, n_recovery)

        self.particles[:, 0] = x_coords
        self.particles[:, 1] = y_coords
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_recovery)

        self.weights = np.ones(n_recovery) / n_recovery

    def update(self, scan_ranges, scan_angle_min, scan_angle_inc, sensor_offset=[0.0, 0.0]):
        """Likelihood Field Model (Numba Optimized)"""
        if self.log_likelihood_map_flat is None:
            return

        if scan_ranges is None:
            return

        n_scans = len(scan_ranges)
        if n_scans == 0:
            return

        if n_scans != self.cached_n_scans:
            self._update_trig_cache(n_scans, scan_angle_min, scan_angle_inc, self.scan_step)

        step = self.scan_step
        raw_ranges = np.array(scan_ranges[::step], dtype=np.float32)

        # 유효한 거리 값만 골라내기
        valid_mask = (raw_ranges > 0.1) & (raw_ranges < 10.0)
        ranges = raw_ranges[valid_mask]

        if ranges.shape[0] == 0:
            return

        # 마스킹 적용
        ranges_cos = self.full_cos_cache[valid_mask]
        ranges_sin = self.full_sin_cache[valid_mask]

        # 센서 데이터 로컬 좌표계로 변환 (Robot Frame)
        laser_x = ranges * ranges_cos + sensor_offset[0]
        laser_y = ranges * ranges_sin + sensor_offset[1]

        # --- Numba Core Function Call (Parallelized) ---
        # 기존 코드의 거대 행렬 생성 부분을 제거하고 JIT 함수 내부 루프로 처리
        total_log_scores = _update_core(
            self.particles,
            laser_x, laser_y,
            self.log_likelihood_map_flat,
            self.map_width, self.map_height, self.map_resolution,
            self.map_origin[0], self.map_origin[1],
            self.penalty_idx
        )

        # 수치 안정성을 위해 max_log_score 사용
        max_log = np.max(total_log_scores)

        # 이전 가중치를 고려하여 업데이트 (Log domain에서 더하기)
        # self.weights는 정규화되어 있으므로 log를 취하거나 곱셈 연산 필요
        prev_weights = self.weights
        # 0인 가중치 방어
        prev_weights = np.maximum(prev_weights, 1e-300)

        log_prev = np.log(prev_weights)
        new_log_weights = log_prev + total_log_scores

        max_log = np.max(new_log_weights)
        weights_unnorm = np.exp(new_log_weights - max_log)

        # 모든 가중치가 0이 되는 경우 방어
        sum_weights = np.sum(weights_unnorm)
        if sum_weights < 1e-15 or np.isnan(sum_weights):
            self._recover_from_kidnapping()
        else:
            self.weights = weights_unnorm / sum_weights

        # 유효 파티클 수 (N_eff) 계산
        n_eff = 1.0 / np.sum(self.weights ** 2)

        # 파티클 수의 절반 이하로 유효 파티클이 떨어졌을 때만 리샘플링
        if n_eff < self.num_particles / 2.0:
            self.resample()

    def resample(self):
        """KLD-Sampling (Numba Optimized)"""
        # Numba 코어 함수 호출로 대체
        new_particles, new_n = _resample_core(
            self.particles,
            self.weights,
            self.min_particles,
            self.max_particles,
            self.kld_err,
            self.kld_z
        )

        # 상태 갱신
        self.num_particles = new_n
        # 버퍼에 결과 복사
        self.particles_buffer[:new_n] = new_particles
        self.particles = self.particles_buffer[:self.num_particles]

        # 가중치 초기화
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def get_estimated_pose(self):
        """Robust Average (Numba Optimized)"""
        if self.num_particles == 0 or np.sum(self.weights) < 1e-9:
            return np.mean(self.particles, axis=0)

        best_idx = np.argmax(self.weights)

        # Numba 코어 함수 호출
        return _get_estimated_pose_core(self.particles, self.weights, best_idx)