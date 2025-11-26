import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# my_particle_filter.py 파일이 같은 디렉토리에 있어야 합니다.
from particle_filter import ParticleFilter


ani = None
# --- 1. Mock ROS Message Classes ---
class Point:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Pose:
    def __init__(self, x, y):
        self.position = Point(x, y)


class MapInfo:
    def __init__(self, width, height, resolution, origin_x, origin_y):
        self.width, self.height, self.resolution = width, height, resolution
        self.origin = Pose(origin_x, origin_y)


class OccupancyGridMsg:
    def __init__(self, data, width, height, res):
        self.info = MapInfo(width, height, res, 0.0, 0.0)
        self.data = data


# --- 2. 시뮬레이션 헬퍼 함수 ---
def create_test_map(width_m, height_m, resolution):
    w_px = int(width_m / resolution)
    h_px = int(height_m / resolution)
    map_grid = np.zeros((h_px, w_px), dtype=np.int8)

    # 외부 테두리 벽
    map_grid[0, :] = 100
    map_grid[-1, :] = 100
    map_grid[:, 0] = 100
    map_grid[:, -1] = 100

    # 내부 장애물 (좌측 상단 방)
    inner_wall_x_m = 3.0
    inner_wall_y_m = 7.0
    ix_px = int(inner_wall_x_m / resolution)
    iy_px = int(inner_wall_y_m / resolution)

    # 인덱스 범위 체크 후 할당
    if 0 <= iy_px < h_px and 0 <= ix_px < w_px:
        map_grid[iy_px:, ix_px] = 100
        map_grid[iy_px, :ix_px + 1] = 100

    flat_data = map_grid.flatten().tolist()
    return OccupancyGridMsg(flat_data, w_px, h_px, resolution), map_grid


def get_true_scan(robot_pose, map_width_m, map_height_m, angles):
    rx, ry, ryaw = robot_pose
    ranges = []
    iw_x = 3.0
    iw_y = 7.0

    for angle in angles:
        global_angle = ryaw + angle
        dx = np.cos(global_angle)
        dy = np.sin(global_angle)
        if abs(dx) < 1e-9: dx = 1e-9 if dx >= 0 else -1e-9
        if abs(dy) < 1e-9: dy = 1e-9 if dy >= 0 else -1e-9

        cand_dists = []
        d1 = (0 - rx) / dx
        d2 = (map_width_m - rx) / dx
        d3 = (0 - ry) / dy
        d4 = (map_height_m - ry) / dy
        cand_dists.extend([d for d in [d1, d2, d3, d4] if d > 0])

        # 내부 장애물 교차 계산
        t_v = (iw_x - rx) / dx
        if t_v > 0:
            y_at_intersect = ry + t_v * dy
            if iw_y <= y_at_intersect <= map_height_m:
                cand_dists.append(t_v)

        t_h = (iw_y - ry) / dy
        if t_h > 0:
            x_at_intersect = rx + t_h * dx
            if 0 <= x_at_intersect <= iw_x:
                cand_dists.append(t_h)

        if cand_dists:
            r = min(cand_dists)
            # 센서 노이즈 추가
            r += np.random.normal(0, 0.02)
            r = np.clip(r, 0.0, 10.0)
        else:
            r = 10.0
        ranges.append(r)
    return np.array(ranges)


def run_simulation():
    map_w_m, map_h_m = 10.0, 10.0
    res = 0.1

    mock_map_msg, map_grid = create_test_map(map_w_m, map_h_m, res)

    # Particle Filter 초기화
    pf = ParticleFilter(min_particles=300, max_particles=3000, initial_noise=[0.1, 0.1, 0.1])
    pf.set_map(mock_map_msg)

    # 시작 위치
    start_x, start_y, start_yaw = 5.0, 2.0, np.pi / 2.0
    true_pose = np.array([start_x, start_y, start_yaw])

    # Global Localization (초기 파티클 랜덤 뿌리기)
    init_num = pf.max_particles
    px = np.random.uniform(0, map_w_m, init_num)
    py = np.random.uniform(0, map_h_m, init_num)
    pyaw = np.random.uniform(-np.pi, np.pi, init_num)
    pf.particles = np.column_stack((px, py, pyaw))
    pf.weights = np.ones(init_num) / init_num
    pf.num_particles = init_num

    # 충돌 감지 헬퍼 함수
    def check_collision(x, y):
        ix = int(x / res)
        iy = int(y / res)
        if ix < 0 or ix >= map_grid.shape[1] or iy < 0 or iy >= map_grid.shape[0]:
            return True
        if map_grid[iy, ix] > 50:
            return True
        return False

    # 그래픽 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, map_w_m)
    ax.set_ylim(0, map_h_m)
    ax.set_title("Autonomous Roaming with Particle Filter")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    if hasattr(pf, 'likelihood_map') and pf.likelihood_map is not None:
        ax.imshow(pf.likelihood_map, origin='lower',
                  extent=[0, map_w_m, 0, map_h_m], cmap='viridis', alpha=0.6)
    else:
        ax.imshow(map_grid, origin='lower',
                  extent=[0, map_w_m, 0, map_h_m], cmap='gray_r', alpha=0.5)

    particle_plot, = ax.plot([], [], 'r.', markersize=2, alpha=0.3, label='Particles')
    robot_plot, = ax.plot([], [], 'bo', markersize=10, markeredgecolor='k', label='True Robot')
    # [수정] Quiver 초기화
    robot_arrow = ax.quiver(start_x, start_y, np.cos(start_yaw), np.sin(start_yaw),
                            color='b', width=0.005, scale=5)
    est_plot, = ax.plot([], [], 'g*', markersize=10, markeredgecolor='k', label='Estimated Pose')
    text_info = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top',
                        bbox=dict(facecolor='white', alpha=0.7))
    ax.legend(loc='lower right')

    # 라이다 설정
    scan_angle_min = -np.pi / 2
    scan_angle_max = np.pi / 2
    scan_angle_inc = np.deg2rad(2.0)
    num_rays = int((scan_angle_max - scan_angle_min) / scan_angle_inc)
    scan_angles = np.linspace(scan_angle_min, scan_angle_max, num_rays)

    # --- 시뮬레이션 상태 변수 ---
    max_frames = 500
    base_speed = 0.15

    def update(frame):
        nonlocal true_pose

        # 1. 이동 계획 (Look Ahead)
        dx_local = 0.0
        dyaw = 0.0

        look_ahead_dist = 0.5
        check_x = true_pose[0] + look_ahead_dist * np.cos(true_pose[2])
        check_y = true_pose[1] + look_ahead_dist * np.sin(true_pose[2])

        if check_collision(check_x, check_y):
            # 벽 감지 시 회전 (Bounce)
            dx_local = 0.0
            bounce_angle = np.random.uniform(np.pi / 2, np.pi)
            direction = np.random.choice([-1, 1])
            dyaw = direction * bounce_angle
        else:
            # 직진
            dx_local = base_speed
            dyaw = np.random.normal(0, 0.05)

        # 2. 로봇 실제 위치 업데이트 (물리 엔진 시뮬레이션)
        # 회전
        true_pose[2] += dyaw
        true_pose[2] = (true_pose[2] + np.pi) % (2 * np.pi) - np.pi

        # 이동 시도
        next_x = true_pose[0] + dx_local * np.cos(true_pose[2])
        next_y = true_pose[1] + dx_local * np.sin(true_pose[2])

        # 실제 충돌 체크 (이동하려는 위치가 벽이면 제자리 정지)
        actual_dx = dx_local
        if not check_collision(next_x, next_y):
            true_pose[0] = next_x
            true_pose[1] = next_y
        else:
            actual_dx = 0.0  # 벽에 막혀 못감

        # 3. 파티클 필터 업데이트
        # [중요] 실제 이동량(actual_dx)을 필터에 전달해야 함
        pf.predict(actual_dx, 0.0, dyaw)

        # 센서 데이터 생성 및 업데이트
        fake_scan = get_true_scan(true_pose, map_w_m, map_h_m, scan_angles)
        pf.update(fake_scan, scan_angle_min, scan_angle_inc)

        est_pose = pf.get_estimated_pose()

        # 4. 시각화 업데이트
        particle_plot.set_data(pf.particles[:, 0], pf.particles[:, 1])
        robot_plot.set_data([true_pose[0]], [true_pose[1]])  # 리스트 감싸기

        # [수정] set_offsets는 [[x, y]] 형태의 2D 배열을 요구함
        robot_arrow.set_offsets(np.array([[true_pose[0], true_pose[1]]]))
        robot_arrow.set_UVC(np.cos(true_pose[2]), np.sin(true_pose[2]))

        if est_pose is not None and not np.isnan(est_pose[0]):
            est_plot.set_data([est_pose[0]], [est_pose[1]])
            dist_error = np.linalg.norm(true_pose[:2] - est_pose[:2])
        else:
            est_plot.set_data([], [])
            dist_error = 99.99

        text_info.set_text(f"Frame: {frame}/{max_frames}\n"
                           f"Particles: {pf.num_particles}\n"
                           f"Error: {dist_error:.2f}m")

        return particle_plot, robot_plot, est_plot, text_info, robot_arrow

    # blit=True는 성능을 높이지만 백엔드에 따라 이슈가 있을 수 있음 (이슈 시 False로 변경)
    ani = FuncAnimation(fig, update, frames=max_frames, blit=False, interval=50)
    plt.show()


if __name__ == "__main__":
    run_simulation()