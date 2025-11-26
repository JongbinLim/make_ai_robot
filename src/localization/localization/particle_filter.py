import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit, prange


# ==============================================================================
# Numba JIT Optimized Kernels (클래스 외부 정의)
# ==============================================================================

@njit(cache=True)
def _fast_predict(particles, dx, dy, dyaw, motion_noise):
    """
    Motion Model 연산 가속
    """
    n = len(particles)
    # 노이즈 생성
    noise_x = np.random.normal(0, motion_noise[0], n)
    noise_y = np.random.normal(0, motion_noise[1], n)
    noise_yaw = np.random.normal(0, motion_noise[2], n)

    # In-place 업데이트를 위한 루프
    for i in range(n):
        p_yaw = particles[i, 2]
        c = np.cos(p_yaw)
        s = np.sin(p_yaw)

        # 노이즈가 섞인 이동량
        d_x_noisy = dx + noise_x[i]
        d_y_noisy = dy + noise_y[i]
        d_yaw_noisy = dyaw + noise_yaw[i]

        # 좌표 변환 및 적용
        particles[i, 0] += (d_x_noisy * c - d_y_noisy * s)
        particles[i, 1] += (d_x_noisy * s + d_y_noisy * c)
        particles[i, 2] += d_yaw_noisy

        # Yaw 정규화 (-pi ~ pi)
        # (angle + pi) % 2pi - pi 방식의 수식
        particles[i, 2] = (particles[i, 2] + np.pi) % (2 * np.pi) - np.pi


@njit(parallel=True, cache=True)
def _fast_update_likelihood(particles,
                            ranges,
                            ranges_cos,
                            ranges_sin,
                            sensor_offset,
                            map_flat,
                            map_width,
                            map_height,
                            map_resolution,
                            map_origin,
                            penalty_idx):
    """
    Likelihood Field Update 가속 (병렬 처리 핵심 구간)
    메모리 할당을 줄이고 CPU 코어를 모두 사용하여 계산합니다.
    """
    n_particles = particles.shape[0]
    n_rays = ranges.shape[0]

    # 결과 점수 배열
    scores = np.zeros(n_particles, dtype=np.float32)

    inv_res = 1.0 / map_resolution
    ox = map_origin[0]
    oy = map_origin[1]

    sensor_x = sensor_offset[0]
    sensor_y = sensor_offset[1]

    # Laser Points in Robot Frame (미리 계산된 cos/sin 사용)
    # ranges는 이미 valid mask가 적용된 상태여야 함
    laser_x = ranges * ranges_cos + sensor_x
    laser_y = ranges * ranges_sin + sensor_y

    # 병렬 루프 (각 파티클은 독립적임)
    for i in prange(n_particles):
        px = particles[i, 0]
        py = particles[i, 1]
        p_yaw = particles[i, 2]

        c = np.cos(p_yaw)
        s = np.sin(p_yaw)

        sum_log_prob = 0.0

        for j in range(n_rays):
            # 1. 로봇 프레임 -> 월드 프레임 변환
            # wx = px + (c * lx - s * ly)
            lx = laser_x[j]
            ly = laser_y[j]

            wx = px + (c * lx - s * ly)
            wy = py + (s * lx + c * ly)

            # 2. 맵 인덱싱
            map_x = int((wx - ox) * inv_res)
            map_y = int((wy - oy) * inv_res)

            idx = penalty_idx  # 기본값: 맵 밖 패널티

            if 0 <= map_x < map_width and 0 <= map_y < map_height:
                idx = map_y * map_width + map_x

            # 3. Log Likelihood 누적
            sum_log_prob += map_flat[idx]

        scores[i] = sum_log_prob

    return scores


@njit(cache=True)
def _fast_resample_kld(particles, weights,
                       min_particles, max_particles,
                       kld_err, kld_z,
                       xy_res, yaw_res):
    """
    KLD Sampling 및 Low Variance Resampling 로직 통합 가속
    """
    # 1. KLD: 현재 분포의 Bin 개수 세기
    n_curr = len(particles)

    # Binning을 위한 큰 정수 multiplier
    m_y = 100000
    m_yaw = 10000000000

    # 3D 좌표 -> 1D 해시 (Binning)
    bins = np.empty(n_curr, dtype=np.int64)
    for i in range(n_curr):
        kx = np.floor(particles[i, 0] / xy_res)
        ky = np.floor(particles[i, 1] / xy_res)
        kyaw = np.floor(particles[i, 2] / yaw_res)
        bins[i] = int(kx) + int(ky) * m_y + int(kyaw) * m_yaw

    # Unique Bin 개수 (k)
    unique_bins = np.unique(bins)
    k = len(unique_bins)

    # KLD 파티클 수 계산 공식
    if k > 1:
        # term1 = 1 - 2/(9(k-1))
        # term2 = sqrt(2/(9(k-1))) * z
        # n = (k-1) / (2*err) * (term1 + term2)^3

        denom = 9.0 * (k - 1)
        term1 = 1.0 - 2.0 / denom
        term2 = np.sqrt(2.0 / denom) * kld_z
        term3 = term1 + term2

        calculated_n = (k - 1) / (2.0 * kld_err) * (term3 * term3 * term3)
        new_n = int(calculated_n)
    else:
        new_n = min_particles

    # 클램핑
    if new_n < min_particles: new_n = min_particles
    if new_n > max_particles: new_n = max_particles

    return new_n


@njit(cache=True)
def _low_variance_sampler(particles, weights, n_resample):
    """
    Low Variance Resampling 알고리즘 (O(N))
    """
    resampled = np.zeros((n_resample, 3), dtype=np.float32)

    # Cumulative Weights
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0 + 1e-6  # 부동소수점 오차 보정

    step = 1.0 / n_resample
    r = np.random.uniform(0, step)

    # 투 포인터 방식 or searchsorted
    # Numba에서는 searchsorted가 매우 빠름
    points = np.arange(n_resample, dtype=np.float32) * step + r
    indices = np.searchsorted(cumsum, points)

    for i in range(n_resample):
        idx = indices[i]
        # 인덱스 범위 안전장치
        if idx >= len(particles):
            idx = len(particles) - 1
        resampled[i] = particles[idx]

    return resampled


# ==============================================================================
# Particle Filter Class
# ==============================================================================

class ParticleFilter:
    def __init__(self,
                 min_particles=300,
                 max_particles=3000,
                 initial_noise=[0.1, 0.1, 0.1]):

        self.min_particles = min_particles
        self.max_particles = max_particles
        self.num_particles = max_particles

        # 버퍼: 메모리 재할당 방지
        self.particles_buffer = np.zeros((self.max_particles, 3), dtype=np.float32)
        self.particles = self.particles_buffer[:self.num_particles]
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

        self.initial_noise = np.array(initial_noise, dtype=np.float32)
        self.motion_noise = np.array([0.02, 0.02, 0.01], dtype=np.float32)

        self.sensor_sigma = 0.1
        self.sensor_model_factor = -0.5 / (self.sensor_sigma ** 2)

        # KLD params
        self.kld_err = 0.05  # 오차 범위 (조금 더 타이트하게 잡음)
        self.kld_z = 2.32  # 99% 신뢰구간 (z값 변경 가능)

        # AMCL params
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.alpha_slow = 0.001
        self.alpha_fast = 0.1

        # Map params
        self.log_likelihood_map_flat = np.zeros(1, dtype=np.float32)
        self.map_resolution = 0.05
        self.map_origin = np.array([0, 0], dtype=np.float32)
        self.map_width = 0
        self.map_height = 0
        self.penalty_idx = 0
        self.free_space_indices = None

        # Caching
        self.cached_n_scans = -1
        self.cached_angle_min = 0.0
        self.cached_angle_inc = 0.0
        self.cached_step = 0
        self.full_cos_cache = None
        self.full_sin_cache = None

        self.scan_step = 5

    def initialize(self, x, y, yaw):
        self.num_particles = self.max_particles
        self.particles = self.particles_buffer[:self.num_particles]

        self.particles[:, 0] = np.random.normal(x, self.initial_noise[0], self.num_particles)
        self.particles[:, 1] = np.random.normal(y, self.initial_noise[1], self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, self.initial_noise[2], self.num_particles)

        # JIT 함수가 아니므로 Numpy 연산 사용
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

        self.w_slow = 0.0
        self.w_fast = 0.0

    def set_map(self, msg):
        """
        Scipy를 사용하는 부분은 JIT 컴파일이 불가능하므로 순수 Python/Numpy로 유지합니다.
        (초기화 시 1회만 실행되므로 성능 영향 적음)
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

        binary_free = (raw_data >= 0) & (raw_data < 50)

        # Random Particle 생성을 위한 캐시
        y_idxs, x_idxs = np.where(binary_free)
        self.free_space_indices = np.column_stack((x_idxs, y_idxs)).astype(np.float32)

        # EDT 및 Log Likelihood 계산
        dist_map_pixels = distance_transform_edt(binary_free)
        dist_map_meters = dist_map_pixels * resolution
        log_likelihood_map = (dist_map_meters ** 2) * self.sensor_model_factor

        min_log_prob = -20.0
        log_likelihood_map = np.maximum(log_likelihood_map, min_log_prob)

        # Flatten 및 패딩 추가
        self.log_likelihood_map_flat = np.append(log_likelihood_map.ravel(), min_log_prob).astype(np.float32)
        self.penalty_idx = self.map_size

    def predict(self, dx, dy, dyaw):
        # Numba JIT 함수 호출
        _fast_predict(self.particles, float(dx), float(dy), float(dyaw), self.motion_noise)

    def _update_trig_cache(self, n_scans, angle_min, angle_inc, step):
        self.cached_n_scans = n_scans
        self.cached_angle_min = angle_min
        self.cached_angle_inc = angle_inc
        self.cached_step = step

        angles = np.arange(n_scans, dtype=np.float32)[::step] * angle_inc + angle_min
        self.full_cos_cache = np.cos(angles).astype(np.float32)
        self.full_sin_cache = np.sin(angles).astype(np.float32)

    def _generate_random_particles(self, n_particles):
        if self.free_space_indices is None or n_particles <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        idx_indices = np.random.choice(len(self.free_space_indices), size=n_particles)
        chosen = self.free_space_indices[idx_indices]

        x = chosen[:, 0] * self.map_resolution + self.map_origin[0]
        y = chosen[:, 1] * self.map_resolution + self.map_origin[1]

        # 그리드 내 랜덤 위치
        x += np.random.uniform(0, self.map_resolution, n_particles)
        y += np.random.uniform(0, self.map_resolution, n_particles)
        yaw = np.random.uniform(-np.pi, np.pi, n_particles)

        return np.column_stack((x, y, yaw)).astype(np.float32)

    def _recover_from_kidnapping(self):
        # 맵 정보가 있을 때만 복구 시도
        if self.free_space_indices is not None:
            self.particles[:] = self._generate_random_particles(self.num_particles)
            self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles
            self.w_slow = 0.0
            self.w_fast = 0.0

    def update(self, scan_ranges, scan_angle_min, scan_angle_inc, sensor_offset=[0.0, 0.0]):
        if self.log_likelihood_map_flat is None or scan_ranges is None:
            return

        n_scans = len(scan_ranges)
        if n_scans == 0: return

        if n_scans != self.cached_n_scans:
            self._update_trig_cache(n_scans, scan_angle_min, scan_angle_inc, self.scan_step)

        # 데이터 전처리 (Numpy)
        step = self.scan_step
        raw_ranges = np.array(scan_ranges[::step], dtype=np.float32)

        # Valid masking
        mask = (raw_ranges > 0.1) & (raw_ranges < 10.0)
        ranges = raw_ranges[mask]

        if ranges.shape[0] == 0:
            return

        ranges_cos = self.full_cos_cache[mask]
        ranges_sin = self.full_sin_cache[mask]

        sensor_offset_np = np.array(sensor_offset, dtype=np.float32)

        # =========================================================
        # [핵심] Numba Accelerated Update
        # 모든 파티클에 대해 Likelihood를 병렬로 계산합니다.
        # =========================================================
        total_log_scores = _fast_update_likelihood(
            self.particles,
            ranges,
            ranges_cos,
            ranges_sin,
            sensor_offset_np,
            self.log_likelihood_map_flat,
            self.map_width,
            self.map_height,
            self.map_resolution,
            self.map_origin,
            self.penalty_idx
        )

        # 후처리 (Python/Numpy)
        max_log = np.max(total_log_scores)
        weights_unnorm = np.exp(total_log_scores - max_log)

        current_w_avg = np.mean(weights_unnorm) * np.exp(max_log)

        if self.w_slow == 0.0:
            self.w_slow = current_w_avg
            self.w_fast = current_w_avg
        else:
            self.w_fast += self.alpha_fast * (current_w_avg - self.w_fast)
            self.w_slow += self.alpha_slow * (current_w_avg - self.w_slow)

        sum_weights = np.sum(weights_unnorm)
        if sum_weights < 1e-15 or np.isnan(sum_weights):
            self._recover_from_kidnapping()
        else:
            self.weights = weights_unnorm / sum_weights

        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2.0:
            self.resample()

    def resample(self):
        # Best Particle 백업
        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx].copy()

        # 1. KLD를 이용해 필요한 파티클 수 계산 (Numba)
        xy_res = 0.1
        yaw_res = np.deg2rad(5.0)

        new_n = _fast_resample_kld(
            self.particles, self.weights,
            self.min_particles, self.max_particles,
            self.kld_err, self.kld_z,
            xy_res, yaw_res
        )

        # 2. Augmented MCL 랜덤 비율 계산
        w_diff = 1.0 - (self.w_fast / self.w_slow)
        random_prob = max(0.0, w_diff)

        num_random = int(new_n * random_prob)
        num_resample = new_n - num_random

        # 3. Low Variance Resampling (Numba)
        if num_resample > 0:
            resampled_particles = _low_variance_sampler(self.particles, self.weights, num_resample)
        else:
            resampled_particles = np.zeros((0, 3), dtype=np.float32)

        # 4. 랜덤 파티클 생성
        if num_random > 0:
            random_particles = self._generate_random_particles(num_random)
        else:
            random_particles = np.zeros((0, 3), dtype=np.float32)

        # 병합
        if len(random_particles) == 0:
            total_particles = resampled_particles
        else:
            total_particles = np.vstack((resampled_particles, random_particles))

        self.num_particles = len(total_particles)
        self.particles_buffer[:self.num_particles] = total_particles
        self.particles = self.particles_buffer[:self.num_particles]

        # Best Particle 복원 (랜덤 노이즈 방지용)
        if num_resample > 0:
            self.particles[0] = best_particle

        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def get_estimated_pose(self):
        # 계산량이 적으므로 Numpy 유지
        if self.num_particles == 0:
            return np.zeros(3)

        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx]

        # Best particle 주변 0.5m 이내만 평균
        dx = self.particles[:, 0] - best_particle[0]
        dy = self.particles[:, 1] - best_particle[1]
        dist_sq = dx * dx + dy * dy

        mask = dist_sq < (0.5 ** 2)
        if np.sum(mask) <= 1:
            return best_particle

        cluster_p = self.particles[mask]
        cluster_w = self.weights[mask]
        cluster_w /= np.sum(cluster_w)

        x = np.sum(cluster_p[:, 0] * cluster_w)
        y = np.sum(cluster_p[:, 1] * cluster_w)

        sin_sum = np.sum(np.sin(cluster_p[:, 2]) * cluster_w)
        cos_sum = np.sum(np.cos(cluster_p[:, 2]) * cluster_w)
        yaw = np.arctan2(sin_sum, cos_sum)

        return np.array([x, y, yaw], dtype=np.float32)