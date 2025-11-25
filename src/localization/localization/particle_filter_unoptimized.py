import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit, prange

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
        self.sensor_sigma = 0.1 # 가우시안 분포의 표준편차 (m)
        # Log 계산을 피하기 위해 미리 상수 계산
        self.sensor_model_factor = -0.5 / (self.sensor_sigma ** 2)

        # KLD 파라미터
        self.kld_err = 0.05 # 오차 허용 범위. 값이 작을수록 정밀해지고 파티클이 많아짐
        self.kld_z = 2.326  # 위에서 설정한 오차범위 안에 실제 분포가 들어올 확률 (z_0.99의 값을 사용). 클수록 안정적이게 되지만 계산량이 늘어남

        # 맵 데이터 저장 변수
        self.log_likelihood_map_flat = None # 최적화를 위해 확률 맵 대신 Log-Likelihood 맵을 저장
        self.map_info = None
        self.map_resolution = 0.05
        self.map_origin = np.array([0, 0], dtype=np.float32)
        self.map_width = 0
        self.map_height = 0

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
        self.particles[:, 0] = np.random.normal(x, self.initial_noise[0], self.num_particles)
        self.particles[:, 1] = np.random.normal(y, self.initial_noise[1], self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, self.initial_noise[2], self.num_particles)

        self._normalize_angles()
        self.weights = np.ones(self.num_particles) / self.num_particles

    def set_map(self, msg):
        """
        ROS OccupancyGrid를 받아 Likelihood Field(거리장)로 변환합니다.
        이 과정은 맵 수신 시 1회만 수행되므로 무겁더라도 update 성능을 위해 투자합니다.
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
        # 0 <= data < 50 인 경우만 확실한 Free Space로 간주
        binary_free = (raw_data >= 0) & (raw_data < 50)

        # 빈 공간 인덱스 캐싱 (랜덤 파티클 생성용)
        y_idxs, x_idxs = np.where(binary_free)
        self.free_space_indices = np.column_stack((x_idxs, y_idxs)).astype(np.float32)

        # EDT 계산 (벽까지의 거리)
        # binary_occupied가 True(1)인 곳은 거리 0, False(0)인 곳은 가장 가까운 True까지의 거리 계산
        # distance_transform_edt는 0인 픽셀에서 가장 가까운 1인 픽셀까지의 거리를 계산하므로 반전 필요
        # 벽(1)을 0으로, 빈공간(0)을 1로 입력해야 빈공간에서 벽까지 거리가 나옴
        dist_map_pixels = distance_transform_edt(binary_free)
        dist_map_meters = dist_map_pixels * resolution

        # 최적화를 위해 exp를 미리 계산하지 않고, Log Likelihood를 바로 저장
        # log(exp(-0.5 * d^2 / sigma^2)) = -0.5 * d^2 / sigma^2
        # 이렇게 하면 update 함수에서 log()와 exp() 연산을 모두 제거 가능
        log_likelihood_map = (dist_map_meters ** 2) * self.sensor_model_factor

        # 맵 밖이나 장애물 내부의 최대 페널티 설정 (log(prob) 값이므로 음수)
        # 너무 작은 값을 주면 underflow가 나므로 적절히 작은 값 설정
        min_log_prob = -20.0  # e^-20 ~= 2e-9
        log_likelihood_map = np.maximum(log_likelihood_map, min_log_prob)

        # 맵 밖 참조를 위한 패딩 전략
        # 맵 데이터 끝에 'min_log_prob' 값을 하나 추가.
        # 인덱스가 맵 밖을 벗어나면, 이 마지막 인덱스를 가리키게 됨
        self.log_likelihood_map_flat = np.append(log_likelihood_map.ravel(), min_log_prob).astype(np.float32)
        self.penalty_idx = self.map_size  # 마지막 인덱스 번호


    def _normalize_angles(self):
        """Yaw를 -pi ~ pi범위로 정규화 시켜줍니다."""
        angles = self.particles[:, 2]
        self.particles[:, 2] = (angles + np.pi) % (2 * np.pi) - np.pi

    def predict(self, dx, dy, dyaw):
        """
        Motion Model: 로봇이 이동한 만큼 파티클들도 이동시킵니다.
        이때 약간의 랜덤 노이즈를 섞어서 파티클을 퍼뜨립니다.
        """
        n = self.num_particles

        # 랜덤 노이즈 생성
        noise = np.random.normal(0, self.motion_noise, (n, 3)).astype(np.float32)

        # 현재 파티클 방향
        p_yaw = self.particles[:, 2]
        c = np.cos(p_yaw)
        s = np.sin(p_yaw)

        # 로봇의 이동량에 노이즈를 더함
        noisy_dx = dx + noise[:, 0]
        noisy_dy = dy + noise[:, 1]
        noisy_dyaw = dyaw + noise[:, 2]

        # 로봇 좌표계(Local) -> 월드 좌표계(Global) 변환 및 이동
        self.particles[:, 0] += (noisy_dx * c - noisy_dy * s)
        self.particles[:, 1] += (noisy_dx * s + noisy_dy * c)
        self.particles[:, 2] += noisy_dyaw

    def _update_trig_cache(self, n_scans, angle_min, angle_inc, step):
        """삼각함수 테이블을 재계산합니다."""
        self.cached_n_scans = n_scans
        self.cached_angle_min = angle_min
        self.cached_angle_inc = angle_inc
        self.cached_step = step

        # 전체 각도 배열 생성
        angles = np.arange(n_scans, dtype=np.float32)[::step] * angle_inc + angle_min

        # sin/cos 미리 계산 (Valid Masking은 update에서 수행)
        self.full_cos_cache = np.cos(angles)
        self.full_sin_cache = np.sin(angles)

    def _recover_from_kidnapping(self):
        """
        모든 파티클이 길을 잃었을 때 수행합니다.
        맵 전체의 Free Space에서 파티클을 무작위로 다시 뿌립니다.
        """
        # 맵의 빈 공간 정보가 없으면 복구 불가 (초기 위치 주변 리셋 혹은 유지)
        if self.free_space_indices is None:
            self.weights = np.ones(self.num_particles) / self.num_particles
            return

        n_recovery = self.num_particles

        # 1. 빈 공간 좌표 중 랜덤하게 n개 선택
        # free_space_indices: [x_idx, y_idx] 형태라고 가정 (set_map 참조)
        num_free_cells = self.free_space_indices.shape[0]

        # 중복을 허용하여 랜덤 인덱스 선택
        rand_indices = np.random.choice(num_free_cells, size=n_recovery)
        chosen_cells = self.free_space_indices[rand_indices]

        # 2. Grid Index -> World Coordinate 변환
        # 픽셀 중심에 약간의 노이즈를 더해 그리드 격자 현상 방지
        x_coords = chosen_cells[:, 0] * self.map_resolution + self.map_origin[0]
        y_coords = chosen_cells[:, 1] * self.map_resolution + self.map_origin[1]

        # 그리드 내 랜덤 위치 미세 조정
        x_coords += np.random.uniform(0, self.map_resolution, n_recovery)
        y_coords += np.random.uniform(0, self.map_resolution, n_recovery)

        # 3. 파티클 위치 덮어쓰기
        self.particles[:, 0] = x_coords
        self.particles[:, 1] = y_coords
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_recovery)  # 방향은 완전 랜덤

        # 4. 가중치 초기화
        self.weights = np.ones(n_recovery) / n_recovery

    def update(self, scan_ranges, scan_angle_min, scan_angle_inc, sensor_offset=[0.0, 0.0]):
        """
        Likelihood Field Model을 이용한 고속 센서 업데이트
        """
        if self.log_likelihood_map_flat is None:
            return

        if scan_ranges is None:
            return

        n_scans = len(scan_ranges)
        if n_scans == 0:
            return

        # LiDAR 데이터 개수가 바뀌지 않았다면 각도 파라미터도 안 바뀌었다고 가정 (빠른 체크)
        # 만약 동적으로 각도 범위가 바뀌는 센서라면 아래 if문에 or 조건으로 추가하면 됨
        if n_scans != self.cached_n_scans:
            self._update_trig_cache(n_scans, scan_angle_min, scan_angle_inc, self.scan_step)

        # 라이다 데이터 다운샘플링 (속도 향상)
        step = self.scan_step # 5개마다 1개씩 사용 (원하는 정확도에 따라 조정 가능)
        raw_ranges = np.array(scan_ranges[::step], dtype=np.float32)

        # 유효한 거리 값만 골라내기
        valid_mask = (raw_ranges > 0.1) & (raw_ranges < 10.0)
        ranges = raw_ranges[valid_mask]

        # 유효한 센서 데이터가 하나도 없으면 업데이트 스킵
        if ranges.shape[0] == 0:
            return

        # 마스킹 적용
        ranges_cos = ranges * self.full_cos_cache[valid_mask]
        ranges_sin = ranges * self.full_sin_cache[valid_mask]

        # 라이다 점들을 로봇 중심 좌표로 변환 (Robot Frame)
        laser_x = ranges_cos + sensor_offset[0]
        laser_y = ranges_sin + sensor_offset[1]

        # 모든 파티클에 대해 라이다 점들을 월드 좌표로 변환 (Vectorization)
        # (N_particles, 1) 형태의 배열 생성
        p_x = self.particles[:, 0][:, np.newaxis]
        p_y = self.particles[:, 1][:, np.newaxis]
        p_yaw = self.particles[:, 2][:, np.newaxis]

        c = np.cos(p_yaw)
        s = np.sin(p_yaw)

        # 회전 변환 + 평행 이동 = 월드 좌표계상의 라이다 점들
        # shape: (num_particles, num_rays)
        wx = p_x + (c * laser_x - s * laser_y)
        wy = p_y + (s * laser_x + c * laser_y)

        # --- 맵 인덱싱 (Likelihood Field Lookup) ---

        # 월드 좌표 -> 맵 그리드 인덱스(x, y) 변환
        inv_res = 1.0 / self.map_resolution
        map_x = np.floor((wx - self.map_origin[0]) * inv_res).astype(np.int32)
        map_y = np.floor((wy - self.map_origin[1]) * inv_res).astype(np.int32)

        # 맵 범위 체크
        in_bounds = (map_x >= 0) & (map_x < self.map_width) & \
                    (map_y >= 0) & (map_y < self.map_height)

        # 1D 인덱스로 변환 (y * width + x)
        flat_indices = map_y * self.map_width + map_x

        flat_indices[~in_bounds] = self.penalty_idx

        # Log-Likelihood Lookup
        # log_scores shape: (N, R)
        log_scores_per_ray = self.log_likelihood_map_flat[flat_indices]
        # 파티클별 총 Log Score 합산 (곱셈 -> 덧셈)
        total_log_scores = np.sum(log_scores_per_ray, axis=1)

        # 수치 안정성을 위해 max_log_score 사용
        max_log = np.max(total_log_scores)

        # 실제 가중치로 변환 (unnormalized)
        weights_unnorm = np.exp(total_log_scores - max_log)

        # 모든 가중치가 0이 되는 경우 방어 (Kidnapped Robot 상황)
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
        """
        KLD-Sampling
        자세한 건 아래 논문 참고
        https://proceedings.neurips.cc/paper_files/paper/2001/file/c5b2cebf15b205503560c4e8e6d1ea78-Paper.pdf
        """
        # 현재 베스트 파티클 백업
        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx].copy()

        # KLD 기반 파티클 수 계산
        # 현재 파티클 분포가 차지하는 '빈(Bin)'의 개수를 세야 함
        # 해상도: 위치 0.2m, 각도 10도 정도로 설정
        xy_res = 0.2
        yaw_res = np.deg2rad(10)

        # 각 파티클의 Bin 인덱스 계산
        k_x = np.floor(self.particles[:, 0] / xy_res).astype(np.int64)
        k_y = np.floor(self.particles[:, 1] / xy_res).astype(np.int64)
        k_yaw = np.floor(self.particles[:, 2] / yaw_res).astype(np.int64)

        # 3D 좌표를 1D 정수(Hash)로 압축 (최적화)
        # x, y, yaw가 겹치지 않도록 충분히 큰 자릿수(Multiplier)를 곱해 더함
        # 예: 맵 크기가 10,000 grid (2km)를 넘지 않는다고 가정 시 100,000이면 충분
        # Python/Numpy는 64비트 정수를 지원하므로 안전함
        bins_flat = k_x + (k_y * 100000) + (k_yaw * 10000000000)

        # 3. 1D Unique 연산 (O(N log N)이지만 1D라 훨씬 빠름)
        unique_bins = np.unique(bins_flat)
        k = len(unique_bins)

        # KLD 공식에 의한 목표 파티클 수 계산
        if k > 1:
            term1 = 1.0 - 2.0 / (9.0 * (k - 1))
            term2 = np.sqrt(2.0 / (9.0 * (k - 1))) * self.kld_z
            term3 = term1 + term2
            new_n = int((k - 1) / (2.0 * self.kld_err) * (term3 ** 3))
        else:
            new_n = self.min_particles

        # 파티클 수 클램핑
        new_n = max(self.min_particles, min(self.max_particles, new_n))
        self.num_particles = new_n

        # 파티클 리샘플링
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0 + 1e-6

        step = 1.0 / new_n
        r = np.random.uniform(0, step)
        points = np.arange(new_n, dtype=np.float32) * step + r

        indices = np.searchsorted(cumsum, points)

        # 선택된 파티클 추출
        chosen_particles = self.particles[indices]

        # 버퍼 업데이트
        self.particles_buffer[:new_n] = chosen_particles

        # 상태 갱신
        self.num_particles = new_n
        self.particles = self.particles_buffer[:self.num_particles]

        # Best Particle은 0번 인덱스에 강제 보존
        self.particles[0] = best_particle

        # 가중치 초기화
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def get_estimated_pose(self):
        """
        단순 가중 평균 대신, 가장 높은 가중치를 가진 파티클 주변의 파티클들만 사용하여
        Robust한 평균을 구합니다. (멀리 떨어진 노이즈 파티클 무시)
        """
        # 가중치 합이 0이거나 파티클이 없으면 안전하게 리턴
        if self.num_particles == 0 or np.sum(self.weights) < 1e-9:
            return np.mean(self.particles, axis=0)

        # 가장 가중치가 높은 파티클 찾기
        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx]

        # Best Particle 주변 일정 반경 내의 파티클만 골라내기
        # 이 반경은 로봇의 크기나 환경에 따라 조절 (보통 0.5m ~ 1.0m)
        search_radius = 1.0

        dx = self.particles[:, 0] - best_particle[0]
        dy = self.particles[:, 1] - best_particle[1]
        # 유클리드 거리 제곱 계산
        dist_sq = dx ** 2 + dy ** 2

        # 반경 내 파티클 마스크 생성
        mask = dist_sq < (search_radius ** 2)

        # 만약 반경 내에 유효한 파티클이 거의 없다면 Best Particle 자체를 반환
        if np.sum(mask) <= 1:
            return best_particle

        # Cluster 내의 파티클과 가중치 추출
        cluster_particles = self.particles[mask]
        cluster_weights = self.weights[mask]

        # 군집 내 가중치 재정규화 (합이 1이 되도록)
        weight_sum = np.sum(cluster_weights)
        if weight_sum < 1e-15:
            return best_particle
        cluster_weights = cluster_weights / weight_sum

        # 군집 내 가중 평균 계산
        x = np.sum(cluster_particles[:, 0] * cluster_weights)
        y = np.sum(cluster_particles[:, 1] * cluster_weights)

        # 각도(Yaw) 평균은 sin, cos 분해해서 계산해야 360도 경계 문제 해결됨
        sin_sum = np.sum(np.sin(cluster_particles[:, 2]) * cluster_weights)
        cos_sum = np.sum(np.cos(cluster_particles[:, 2]) * cluster_weights)
        yaw = np.arctan2(sin_sum, cos_sum)

        return np.array([x, y, yaw])
