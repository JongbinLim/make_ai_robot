import numpy as np
from scipy.ndimage import distance_transform_edt
import sys
from numba import njit


# ==============================================================================
# Numba Optimized Kernels (JIT 컴파일 함수들)
# ==============================================================================

@njit(fastmath=True, cache=True)
def predict_kernel(particles, dx, dy, dyaw, alphas, min_noise):
    """
    Motion Model 커널
    Numpy 벡터 연산 대신 명시적 루프를 사용할 수도 있지만,
    여기서는 기본 벡터 연산이 Numba에서도 매우 효율적으로 최적화됩니다.
    """
    n = len(particles)

    # 이동 거리 및 회전량 계산
    dist_trans = np.sqrt(dx ** 2 + dy ** 2)
    dist_rot = np.abs(dyaw)

    # 노이즈 표준편차 계산
    sigma_x = max(alphas[0] * dist_trans + alphas[1] * dist_rot, min_noise[0])
    sigma_y = max(alphas[2] * dist_trans + alphas[3] * dist_rot, min_noise[1])
    sigma_yaw = max(alphas[4] * dist_trans + alphas[5] * dist_rot, min_noise[2])

    # 파티클 업데이트 루프
    # Numba는 루프 내부의 sin/cos 연산을 매우 효율적으로 병렬화/최적화합니다.
    for i in range(n):
        # 노이즈 생성
        noise_x = np.random.normal(0, sigma_x)
        noise_y = np.random.normal(0, sigma_y)
        noise_yaw = np.random.normal(0, sigma_yaw)

        noisy_dx = dx + noise_x
        noisy_dy = dy + noise_y
        noisy_dyaw = dyaw + noise_yaw

        p_x = particles[i, 0]
        p_y = particles[i, 1]
        p_yaw = particles[i, 2]

        c = np.cos(p_yaw)
        s = np.sin(p_yaw)

        # 위치 업데이트
        particles[i, 0] = p_x + (noisy_dx * c - noisy_dy * s)
        particles[i, 1] = p_y + (noisy_dx * s + noisy_dy * c)
        particles[i, 2] = p_yaw + noisy_dyaw


@njit(fastmath=True, cache=True)
def update_likelihood_kernel(particles, ranges, ranges_cos, ranges_sin,
                             map_flat, dist_map_flat,
                             map_width, map_height, map_res, map_origin_x, map_origin_y,
                             sensor_offset_x, sensor_offset_y,
                             penalty_idx, dist_threshold, min_log_prob):
    """
    Sensor Model 커널 (가장 부하가 큰 부분)
    모든 파티클 * 모든 레이에 대해 좌표 변환 및 맵 조회를 수행합니다.
    """
    n_particles = len(particles)
    n_rays = len(ranges)
    inv_res = 1.0 / map_res

    weights_unnorm = np.zeros(n_particles, dtype=np.float32)

    # 동적 장애물 처리를 위한 페널티 상수
    penalty_for_dynamic = -1.5

    for i in range(n_particles):
        px = particles[i, 0]
        py = particles[i, 1]
        pyaw = particles[i, 2]

        c = np.cos(pyaw)
        s = np.sin(pyaw)

        total_log_score = 0.0

        for j in range(n_rays):
            # 1. 로봇 프레임 내 레이 좌표
            # 미리 계산된 ranges_cos/sin 사용
            r_cos = ranges_cos[j]
            r_sin = ranges_sin[j]

            # 센서 오프셋 적용
            lx = r_cos + sensor_offset_x
            ly = r_sin + sensor_offset_y

            # 2. 월드 좌표 변환
            wx = px + (c * lx - s * ly)
            wy = py + (s * lx + c * ly)

            # 3. 맵 인덱스 계산
            mx = int((wx - map_origin_x) * inv_res)
            my = int((wy - map_origin_y) * inv_res)

            idx = penalty_idx  # 기본값: 맵 밖

            # 맵 범위 체크
            if 0 <= mx < map_width and 0 <= my < map_height:
                idx = my * map_width + mx

            # 4. 점수 조회
            score = map_flat[idx]
            dist_val = dist_map_flat[idx]

            # 5. 동적 장애물 처리 (Robust Likelihood)
            # 센서값은 유효한데 맵 상의 거리가 멀다면(동적 장애물), 페널티 완화
            if dist_val > dist_threshold:
                if score < penalty_for_dynamic:
                    score = penalty_for_dynamic

            total_log_score += score

        # Log-Sum-Exp 트릭을 위해 여기서는 log score만 저장
        weights_unnorm[i] = total_log_score

    # Log-Sum-Exp trick to avoid overflow/underflow
    max_log = -1e9  # 매우 작은 수
    for i in range(n_particles):
        if weights_unnorm[i] > max_log:
            max_log = weights_unnorm[i]

    sum_w = 0.0
    for i in range(n_particles):
        # exp(score - max)
        val = np.exp(weights_unnorm[i] - max_log)
        weights_unnorm[i] = val
        sum_w += val

    return weights_unnorm, sum_w


@njit(fastmath=True, cache=True)
def compute_kld_number_kernel(particles, weights, xy_res, yaw_res, kld_z, kld_err, min_particles, max_particles):
    """
    KLD(Kullback-Leibler Divergence) 샘플링 수 계산 커널
    """
    n = len(particles)

    # Bin 계산을 위한 간단한 해싱
    # 3D 좌표 -> 1D 해시
    # 정수 오버플로우 방지를 위해 x, y 범위가 너무 크지 않다고 가정하거나 배율 조정

    # 단순화를 위해 정렬을 이용한 유니크 카운팅
    bins = np.empty(n, dtype=np.int64)

    for i in range(n):
        kx = np.int64(np.floor(particles[i, 0] / xy_res))
        ky = np.int64(np.floor(particles[i, 1] / xy_res))
        kyaw = np.int64(np.floor(particles[i, 2] / yaw_res))

        # 해시 충돌 가능성이 매우 낮은 승수 사용
        bins[i] = kx + ky * 100000 + kyaw * 10000000000

    # 정렬 후 유니크 개수 세기 (np.unique보다 빠를 수 있음)
    bins.sort()

    k = 0
    if n > 0:
        k = 1
        for i in range(1, n):
            if bins[i] != bins[i - 1]:
                k += 1

    if k > 1:
        term1 = 1.0 - 2.0 / (9.0 * (k - 1))
        term2 = np.sqrt(2.0 / (9.0 * (k - 1))) * kld_z
        term3 = term1 + term2
        new_n = int((k - 1) / (2.0 * kld_err) * (term3 ** 3))
    else:
        new_n = min_particles

    # 클램핑 (Python max/min은 Numba에서 지원)
    if new_n < min_particles: new_n = min_particles
    if new_n > max_particles: new_n = max_particles

    return new_n


@njit(fastmath=True, cache=True)
def low_variance_resample_kernel(particles, weights, new_n, best_particle):
    """
    Low Variance Resampling (Wheel Algorithm)
    O(N) 복잡도로 매우 빠름
    """
    n_current = len(weights)
    new_particles = np.zeros((new_n, 3), dtype=np.float32)

    # Best particle 보존
    new_particles[0] = best_particle

    step = 1.0 / new_n
    r = np.random.uniform(0, step)

    c = weights[0]
    i = 0

    # 1번째 파티클은 이미 채웠으므로 1부터 시작
    for j in range(1, new_n):
        u = r + j * step
        while u > c and i < n_current - 1:
            i += 1
            c += weights[i]

        new_particles[j, 0] = particles[i, 0]
        new_particles[j, 1] = particles[i, 1]
        new_particles[j, 2] = particles[i, 2]

    return new_particles


@njit(fastmath=True, cache=True)
def compute_pose_mean_kernel(particles, weights, top_k_ratio):
    """
    상위 K% 파티클 기반 가중 평균 계산
    """
    n = len(particles)
    n_top = int(max(5, n * top_k_ratio))

    # Numba에서는 argsort가 지원됨
    # 가중치 내림차순 정렬 인덱스
    # (참고: 전체 정렬은 O(NlogN)이지만 N=3000 정도에서는 충분히 빠름)
    idxs = np.argsort(weights)[::-1]

    top_idxs = idxs[:n_top]

    x_sum = 0.0
    y_sum = 0.0
    sin_sum = 0.0
    cos_sum = 0.0
    w_sum = 0.0

    for i in range(n_top):
        idx = top_idxs[i]
        w = weights[idx]

        x_sum += particles[idx, 0] * w
        y_sum += particles[idx, 1] * w
        sin_sum += np.sin(particles[idx, 2]) * w
        cos_sum += np.cos(particles[idx, 2]) * w
        w_sum += w

    if w_sum == 0:
        return np.mean(particles[:, 0]), np.mean(particles[:, 1]), np.mean(particles[:, 2])

    return x_sum / w_sum, y_sum / w_sum, np.arctan2(sin_sum, cos_sum)


@njit(fastmath=True, cache=True)
def normalize_angle_kernel(particles):
    """각도 정규화 (-pi ~ pi)"""
    n = len(particles)
    for i in range(n):
        angle = particles[i, 2]
        particles[i, 2] = (angle + np.pi) % (2 * np.pi) - np.pi


# ==============================================================================
# Main Class
# ==============================================================================

class ParticleFilter:
    def __init__(self,
                 min_particles=300,
                 max_particles=3000,
                 initial_noise=[0.1, 0.1, 0.1]):

        self.min_particles = min_particles
        self.max_particles = max_particles
        self.num_particles = max_particles

        # Numba 호환성을 위해 Contiguous Array 유지
        self.particles = np.zeros((self.max_particles, 3), dtype=np.float32)
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

        self.initial_noise = np.array(initial_noise, dtype=np.float32)

        # Motion Parameters
        #self.motion_alphas = np.array([0.1, 0.1, 0.05, 0.1, 0.07, 0.1], dtype=np.float32)
        self.motion_alphas = np.array([0.9, 0.9, 0.04, 0.09, 0.06, 0.09], dtype=np.float32)
        self.min_motion_noise = np.array([0.01, 0.01, 0.015], dtype=np.float32)

        self.last_estimated_pose = None

        # Sensor Model Parameters
        self.sensor_sigma = 0.18
        self.sensor_model_factor = -0.5 / (self.sensor_sigma ** 2)
        self.dist_threshold = self.sensor_sigma * 3

        # KLD Parameters
        self.kld_err = 0.015
        self.kld_z = 2.326

        # Map Data
        self.log_likelihood_map_flat = None
        self.dist_map_flat = None
        self.map_info = None
        self.map_resolution = 0.025
        self.map_origin = np.array([0, 0], dtype=np.float32)
        self.map_width = 0
        self.map_height = 0
        self.penalty_idx = 0

        self.free_space_indices = None

        # Trig Cache
        self.cached_n_scans = -1
        self.full_sin_cache = None
        self.full_cos_cache = None
        self.scan_step = 4

    def initialize(self, x, y, yaw):
        self.num_particles = self.max_particles
        # 버퍼 전체가 아니라 현재 필요한 만큼만 슬라이싱하여 사용
        # (주의: Numba 함수에 넘길 때는 항상 실제 데이터 크기에 맞춰야 함)

        # 초기화는 자주 일어나지 않으므로 일반 Numpy 사용
        self.particles[:self.num_particles, 0] = np.random.normal(x, self.initial_noise[0], self.num_particles)
        self.particles[:self.num_particles, 1] = np.random.normal(y, self.initial_noise[1], self.num_particles)
        self.particles[:self.num_particles, 2] = np.random.normal(yaw, self.initial_noise[2], self.num_particles)

        normalize_angle_kernel(self.particles[:self.num_particles])
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def set_map(self, msg):
        """
        맵 처리는 초기 1회만 수행되므로 Numba 최적화에서 제외하거나(Scipy 의존성),
        필요하다면 별도로 처리. 여기서는 기존 로직 유지.
        """
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if width <= 0 or height <= 0: return

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        raw_data = np.array(msg.data, dtype=np.int8).reshape(height, width)
        self.map_resolution = resolution
        self.map_origin = np.array([origin_x, origin_y], dtype=np.float32)
        self.map_width = width
        self.map_height = height
        self.map_size = width * height

        binary_free = (raw_data >= 0) & (raw_data < 50)

        # 빈 공간 인덱싱
        y_idxs, x_idxs = np.where(binary_free)
        self.free_space_indices = np.column_stack((x_idxs, y_idxs)).astype(np.float32)

        # EDT 계산
        dist_map_pixels = distance_transform_edt(binary_free)
        dist_map_meters = dist_map_pixels * resolution

        log_likelihood_map = (dist_map_meters ** 2) * self.sensor_model_factor
        min_log_prob = -10.0
        log_likelihood_map = np.maximum(log_likelihood_map, min_log_prob)

        # Flat Map 생성
        self.log_likelihood_map_flat = np.append(log_likelihood_map.ravel(), min_log_prob).astype(np.float32)
        self.dist_map_flat = np.append(dist_map_meters.ravel(), 0.0).astype(np.float32)
        self.penalty_idx = self.map_size

    def predict(self, dx, dy, dyaw):
        """Numba Kernel 호출"""
        # 현재 활성화된 파티클 슬라이스 전달
        active_particles = self.particles[:self.num_particles]

        predict_kernel(
            active_particles,
            float(dx), float(dy), float(dyaw),
            self.motion_alphas, self.min_motion_noise
        )

        # 각도 정규화
        normalize_angle_kernel(active_particles)

    def _update_trig_cache(self, n_scans, angle_min, angle_inc, step):
        self.cached_n_scans = n_scans
        angles = np.arange(n_scans, dtype=np.float32)[::step] * angle_inc + angle_min
        self.full_cos_cache = np.cos(angles).astype(np.float32)
        self.full_sin_cache = np.sin(angles).astype(np.float32)

    def _recover_from_kidnapping(self):
        """납치 복구 로직 (기존 Python 로직 유지)"""
        if self.free_space_indices is None: return

        print("[PF] Kidnap detected. Injecting random particles.")

        keep_ratio = 0.3
        n_keep = int(self.num_particles * keep_ratio)
        n_random = self.num_particles - n_keep

        # 상위 파티클 유지는 argsort로
        sorted_indices = np.argsort(self.weights)[::-1]

        # Numba 호환성을 위해 버퍼 직접 조작 대신 복사 방식 사용
        # 현재 활성 파티클
        current_p = self.particles[:self.num_particles]

        # 상위 n_keep개
        kept_particles = current_p[sorted_indices[:n_keep]].copy()

        # 랜덤 파티클 생성
        num_free = self.free_space_indices.shape[0]
        rand_indices = np.random.choice(num_free, size=n_random)
        chosen = self.free_space_indices[rand_indices]

        rand_particles = np.zeros((n_random, 3), dtype=np.float32)
        rand_particles[:, 0] = chosen[:, 0] * self.map_resolution + self.map_origin[0]
        rand_particles[:, 1] = chosen[:, 1] * self.map_resolution + self.map_origin[1]
        rand_particles[:, 0] += np.random.uniform(0, self.map_resolution, n_random)
        rand_particles[:, 1] += np.random.uniform(0, self.map_resolution, n_random)
        rand_particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_random)

        # 병합
        self.particles[:n_keep] = kept_particles
        self.particles[n_keep:self.num_particles] = rand_particles

        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def update(self, scan_ranges, scan_angle_min, scan_angle_inc, sensor_offset=[0.0, 0.0]):
        if self.log_likelihood_map_flat is None: return
        if scan_ranges is None or len(scan_ranges) == 0: return

        n_scans = len(scan_ranges)
        if n_scans != self.cached_n_scans:
            self._update_trig_cache(n_scans, scan_angle_min, scan_angle_inc, self.scan_step)

        # 데이터 전처리
        step = self.scan_step
        raw_ranges = np.array(scan_ranges[::step], dtype=np.float32)

        # Valid Masking
        valid_mask = (raw_ranges > 0.01) & (raw_ranges < 20.0)
        ranges = raw_ranges[valid_mask]

        if len(ranges) == 0: return

        # 마스킹된 cos/sin 배열 준비
        ranges_cos = (ranges * self.full_cos_cache[valid_mask]).astype(np.float32)
        ranges_sin = (ranges * self.full_sin_cache[valid_mask]).astype(np.float32)

        # === Numba Kernel 호출 ===
        # 무거운 연산(좌표변환 + 맵조회 + 확률계산)을 모두 넘김
        weights_unnorm, sum_weights = update_likelihood_kernel(
            self.particles[:self.num_particles],
            ranges, ranges_cos, ranges_sin,
            self.log_likelihood_map_flat, self.dist_map_flat,
            self.map_width, self.map_height, self.map_resolution,
            self.map_origin[0], self.map_origin[1],
            float(sensor_offset[0]), float(sensor_offset[1]),
            self.penalty_idx, self.dist_threshold, -10.0
        )

        # Kidnapped 체크
        if sum_weights < 1e-15 or np.isnan(sum_weights):
            self._recover_from_kidnapping()
        else:
            self.weights = weights_unnorm / sum_weights

        # Resampling Check
        n_eff = 1.0 / np.sum(self.weights ** 2)

        resampled = False
        if n_eff < self.num_particles / 2.0:
            self.resample()
            resampled = True

        # Debug Print (Optional)
        # sys.stdout.write(f"\rN: {self.num_particles} Neff: {n_eff:.1f} MaxW: {np.max(self.weights):.4f}")
        # sys.stdout.flush()

    def resample(self):
        """KLD + Low Variance Resampling using Numba"""
        # 1. KLD로 목표 파티클 수 계산
        new_n = compute_kld_number_kernel(
            self.particles[:self.num_particles],
            self.weights,
            0.1, np.deg2rad(2),  # resolution params
            self.kld_z, self.kld_err,
            self.min_particles, self.max_particles
        )

        # Best particle 찾기 (Python side)
        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx].copy()

        # 2. Low Variance Resampling 실행
        new_particles = low_variance_resample_kernel(
            self.particles[:self.num_particles],
            self.weights,
            new_n,
            best_particle
        )

        # 상태 업데이트
        self.num_particles = new_n
        # 버퍼에 복사
        self.particles[:new_n] = new_particles
        self.weights = np.ones(new_n, dtype=np.float32) / new_n

    def get_estimated_pose(self):
        """Numba 커널을 이용해 가중 평균 계산"""
        if self.num_particles == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        x, y, yaw = compute_pose_mean_kernel(
            self.particles[:self.num_particles],
            self.weights,
            0.2  # top 20%
        )

        current_pose = np.array([x, y, yaw], dtype=np.float32)
        self.last_estimated_pose = current_pose
        return current_pose
