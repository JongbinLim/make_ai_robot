#include "particle_filter.hpp"
#include <queue>
#include <omp.h>
#include <cstring> // For memcpy

// 상수 정의
constexpr float PI = 3.14159265359f;
constexpr float TWO_PI = 2.0f * PI;

// 캐시 지역성을 높이기 위해 계산에 필요한 데이터만 모은 구조체
struct PrecomputedPoint {
    float x;
    float y;
};

ParticleFilter::ParticleFilter(int min_particles, int max_particles,
                               float init_noise_x, float init_noise_y, float init_noise_yaw)
    : min_particles_(min_particles), max_particles_(max_particles),
      map_width_(0), map_height_(0),
      sensor_sigma_(0.25f), kld_err_(0.015f), kld_z_(2.326f) {

    init_noise_[0] = init_noise_x;
    init_noise_[1] = init_noise_y;
    init_noise_[2] = init_noise_yaw;

    float alphas[] = {0.05f, 0.05f, 0.03f, 0.05f, 0.03f, 0.05f};
    std::memcpy(motion_alphas_, alphas, sizeof(alphas));

    min_motion_noise_[0] = 0.02f;
    min_motion_noise_[1] = 0.02f;
    min_motion_noise_[2] = 0.02f;

    sensor_model_factor_ = -0.5f / (sensor_sigma_ * sensor_sigma_);
    dist_threshold_ = sensor_sigma_ * 3.0f;

    std::random_device rd;
    gen_ = std::mt19937(rd());
}

ParticleFilter::~ParticleFilter() {}

inline float ParticleFilter::normalize_angle(float angle) {
    while (angle > PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

void ParticleFilter::initialize(float x, float y, float yaw) {
    particles_.resize(max_particles_);

    std::normal_distribution<float> dist_x(x, init_noise_[0]);
    std::normal_distribution<float> dist_y(y, init_noise_[1]);
    std::normal_distribution<float> dist_yaw(yaw, init_noise_[2]);

    float initial_weight = 1.0f / max_particles_;

    for (auto& p : particles_) {
        p.x = dist_x(gen_);
        p.y = dist_y(gen_);
        p.yaw = normalize_angle(dist_yaw(gen_));
        p.weight = initial_weight;
    }
}

void ParticleFilter::set_map(const std::vector<int8_t>& map_data, int width, int height,
                             float resolution, float origin_x, float origin_y) {
    map_width_ = width;
    map_height_ = height;
    map_resolution_ = resolution;
    map_origin_x_ = origin_x;
    map_origin_y_ = origin_y;

    int size = width * height;
    dist_map_.assign(size, std::numeric_limits<float>::max());
    log_likelihood_map_.resize(size);
    free_space_indices_.clear();

    std::queue<int> q;
    std::vector<bool> visited(size, false);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int8_t val = map_data[idx];

            if (val >= 50) {
                dist_map_[idx] = 0.0f;
                q.push(idx);
                visited[idx] = true;
            } else if (val >= 0) {
                free_space_indices_.push_back({x, y});
            }
        }
    }

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        int curr_idx = q.front();
        q.pop();

        int cx = curr_idx % width;
        int cy = curr_idx / width;

        float curr_dist = dist_map_[curr_idx];

        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                if (!visited[n_idx]) {
                    dist_map_[n_idx] = curr_dist + resolution;
                    visited[n_idx] = true;
                    q.push(n_idx);
                }
            }
        }
    }

    float min_log_prob = -10.0f;
    for (int i = 0; i < size; ++i) {
        if (dist_map_[i] == std::numeric_limits<float>::max()) {
            dist_map_[i] = 100.0f;
        }
        float val = (dist_map_[i] * dist_map_[i]) * sensor_model_factor_;
        log_likelihood_map_[i] = std::max(val, min_log_prob);
    }
}

void ParticleFilter::predict(float dx, float dy, float dyaw) {
    float dist_trans = std::sqrt(dx*dx + dy*dy);
    float dist_rot = std::abs(dyaw);

    float sigma_x = std::max(motion_alphas_[0]*dist_trans + motion_alphas_[1]*dist_rot, min_motion_noise_[0]);
    float sigma_y = std::max(motion_alphas_[2]*dist_trans + motion_alphas_[3]*dist_rot, min_motion_noise_[1]);
    float sigma_yaw = std::max(motion_alphas_[4]*dist_trans + motion_alphas_[5]*dist_rot, min_motion_noise_[2]);

    #pragma omp parallel
    {
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num());
        std::normal_distribution<float> noise_x(0.0f, sigma_x);
        std::normal_distribution<float> noise_y(0.0f, sigma_y);
        std::normal_distribution<float> noise_yaw(0.0f, sigma_yaw);

        #pragma omp for
        for (size_t i = 0; i < particles_.size(); ++i) {
            float nx = noise_x(local_gen);
            float ny = noise_y(local_gen);
            float nyaw = noise_yaw(local_gen);

            float c = std::cos(particles_[i].yaw);
            float s = std::sin(particles_[i].yaw);

            // Rotate motion into map frame
            float noisy_dx = dx + nx;
            float noisy_dy = dy + ny;

            particles_[i].x += (noisy_dx * c - noisy_dy * s);
            particles_[i].y += (noisy_dx * s + noisy_dy * c);
            particles_[i].yaw = normalize_angle(particles_[i].yaw + dyaw + nyaw);
        }
    }
}

void ParticleFilter::update_trig_cache(int n_scans, float angle_min, float angle_inc) {
    if (n_scans == cached_num_scans_) return;

    cached_num_scans_ = n_scans;
    cos_cache_.resize(n_scans);
    sin_cache_.resize(n_scans);
    for (int i = 0; i < n_scans; ++i) {
        float angle = angle_min + i * angle_inc;
        cos_cache_[i] = std::cos(angle);
        sin_cache_[i] = std::sin(angle);
    }
}

void ParticleFilter::update(const std::vector<float>& scan_ranges, float angle_min, float angle_inc, const float sensor_offset[2]) {
    if (log_likelihood_map_.empty()) return;

    int n_scans = scan_ranges.size();
    int step = 4; // Downsampling

    // 캐시 업데이트
    update_trig_cache(n_scans, angle_min, angle_inc);

    float inv_res = 1.0f / map_resolution_;
    float min_log_prob = -10.0f;
    float penalty_for_dynamic = -1.5f;

    // --- 로봇 프레임 기준 유효 센서 포인트 선계산 (Loop Hoisting) ---
    static std::vector<PrecomputedPoint> active_points;
    active_points.clear();
    active_points.reserve(n_scans / step + 1);

    for (int j = 0; j < n_scans; j += step) {
        float r = scan_ranges[j];
        if (r < 0.01f || r > 20.0f) continue;

        // 로봇 프레임 좌표 + 센서 오프셋까지 미리 계산
        float lx = r * cos_cache_[j] + sensor_offset[0];
        float ly = r * sin_cache_[j] + sensor_offset[1];
        active_points.push_back({lx, ly});
    }

    size_t num_particles = particles_.size();
    size_t num_active_scans = active_points.size();

    std::vector<float> log_weights(num_particles);
    float max_log_w = -1e15f;

    // 포인터 캐싱 (멤버 변수 접근 오버헤드 감소)
    const float* likelihood_map_ptr = log_likelihood_map_.data();
    const float* dist_map_ptr = dist_map_.data();
    int m_width = map_width_;
    int m_height = map_height_;
    float m_origin_x = map_origin_x_;
    float m_origin_y = map_origin_y_;
    float threshold = dist_threshold_;

    #pragma omp parallel for reduction(max: max_log_w) schedule(static)
    for (size_t i = 0; i < num_particles; ++i) {
        Particle& p = particles_[i];
        float px = p.x;
        float py = p.y;

        // 파티클의 방향에 대한 sin/cos는 파티클당 1회만 계산
        float s, c;
        s = std::sin(p.yaw);
        c = std::cos(p.yaw);

        float total_log_score = 0.0f;

        for (size_t k = 0; k < num_active_scans; ++k) {
            const auto& pt = active_points[k];

            // 월드 좌표 변환: Rotation + Translation
            float wx = px + (c * pt.x - s * pt.y);
            float wy = py + (s * pt.x + c * pt.y);

            // 좌표를 정수 인덱스로 변환 (Fast Cast)
            int mx = static_cast<int>((wx - m_origin_x) * inv_res);
            int my = static_cast<int>((wy - m_origin_y) * inv_res);

            float score = min_log_prob;

            // 경계 검사
            if (mx >= 0 && mx < m_width && my >= 0 && my < m_height) {
                int idx = my * m_width + mx;

                // 메모리 접근
                score = likelihood_map_ptr[idx];
                float dist = dist_map_ptr[idx];

                if (dist > threshold && score < penalty_for_dynamic) {
                    score = penalty_for_dynamic;
                }
            }
            total_log_score += score;
        }
        log_weights[i] = total_log_score;
        if (total_log_score > max_log_w) max_log_w = total_log_score;
    }

    // Weight Normalization
    float sum_w = 0.0f;
    for (size_t i = 0; i < num_particles; ++i) {
        float w = std::exp(log_weights[i] - max_log_w);
        particles_[i].weight = w;
        sum_w += w;
    }

    if (sum_w < 1e-9f || std::isnan(sum_w)) {
        recover_from_kidnapping();
    } else {
        float inv_sum_w = 1.0f / sum_w;
        float sum_sq_w = 0.0f;
        for (auto& p : particles_) {
            p.weight *= inv_sum_w;
            sum_sq_w += p.weight * p.weight;
        }

        if (1.0f / sum_sq_w < num_particles / 1.5f) {
            resample();
        }
    }
}

void ParticleFilter::resample() {
    float xy_res = 0.1f;
    float yaw_res = 2.0f * PI / 180.0f;

    std::vector<long> bins;
    bins.reserve(particles_.size());
    for(const auto& p : particles_) {
        long kx = static_cast<long>(p.x / xy_res);
        long ky = static_cast<long>(p.y / xy_res);
        long kyaw = static_cast<long>(p.yaw / yaw_res);
        bins.push_back(kx ^ (ky << 10) ^ (kyaw << 20));
    }
    std::sort(bins.begin(), bins.end());
    int k = std::unique(bins.begin(), bins.end()) - bins.begin();

    int new_n = min_particles_;
    if (k > 1) {
        float k_minus_1 = (float)(k - 1);
        float term1 = 1.0f - 2.0f / (9.0f * k_minus_1);
        float term2 = std::sqrt(2.0f / (9.0f * k_minus_1)) * kld_z_;
        float term3 = term1 + term2;
        new_n = static_cast<int>(k_minus_1 / (2.0f * kld_err_) * (term3 * term3 * term3));
    }
    new_n = std::max(min_particles_, std::min(new_n, max_particles_));

    std::vector<Particle> new_particles;
    new_particles.reserve(new_n);

    // Elitism: 가장 좋은 파티클은 무조건 유지
    auto best_it = std::max_element(particles_.begin(), particles_.end(),
        [](const Particle& a, const Particle& b){ return a.weight < b.weight; });
    new_particles.push_back(*best_it);
    new_particles.back().weight = 1.0f / new_n; // 가중치 초기화

    float step = 1.0f / new_n;
    std::uniform_real_distribution<float> dist(0.0f, step);
    float r = dist(gen_);

    float c = particles_[0].weight;
    int idx = 0;
    int particles_size = particles_.size(); // 캐싱

    // Elitism으로 1개 넣었으므로 new_n-1개만 더 뽑음
    for (int i = 1; i < new_n; ++i) {
        float u = r + (float)(i-1) * step; // i-1로 보정
        while (u > c && idx < particles_size - 1) {
            idx++;
            c += particles_[idx].weight;
        }
        Particle p = particles_[idx];
        p.weight = 1.0f / new_n;
        new_particles.push_back(p);
    }

    particles_ = std::move(new_particles);
}

void ParticleFilter::recover_from_kidnapping() {
    if (free_space_indices_.empty()) return;

    int n_keep = static_cast<int>(particles_.size() * 0.3);
    int n_random = particles_.size() - n_keep;

    std::sort(particles_.begin(), particles_.end(),
              [](const Particle& a, const Particle& b) { return a.weight > b.weight; });

    particles_.resize(n_keep + n_random);

    std::uniform_int_distribution<int> idx_dist(0, free_space_indices_.size() - 1);
    std::uniform_real_distribution<float> offset_dist(0.0f, map_resolution_);
    std::uniform_real_distribution<float> yaw_dist(-PI, PI);

    for (size_t i = n_keep; i < particles_.size(); ++i) {
        int idx = idx_dist(gen_);
        auto coord = free_space_indices_[idx];

        particles_[i].x = coord.first * map_resolution_ + map_origin_x_ + offset_dist(gen_);
        particles_[i].y = coord.second * map_resolution_ + map_origin_y_ + offset_dist(gen_);
        particles_[i].yaw = yaw_dist(gen_);
        particles_[i].weight = 1.0f / particles_.size();
    }
}

std::vector<float> ParticleFilter::get_estimated_pose() {
    if (particles_.empty()) return {0,0,0};

    std::vector<int> indices(particles_.size());
    std::iota(indices.begin(), indices.end(), 0);

    int n_top = std::max(5, (int)(particles_.size() * 0.2));
    std::partial_sort(indices.begin(), indices.begin() + n_top, indices.end(),
        [&](int i, int j){ return particles_[i].weight > particles_[j].weight; });

    float x_sum = 0.0f;
    float y_sum = 0.0f;
    float sin_sum = 0.0f;
    float cos_sum = 0.0f;
    float w_sum = 0.0f;

    for (int i = 0; i < n_top; ++i) {
        const auto& p = particles_[indices[i]];
        x_sum += p.x * p.weight;
        y_sum += p.y * p.weight;
        sin_sum += std::sin(p.yaw) * p.weight;
        cos_sum += std::cos(p.yaw) * p.weight;
        w_sum += p.weight;
    }

    if (w_sum == 0) return {0,0,0};
    return {x_sum / w_sum, y_sum / w_sum, std::atan2(sin_sum, cos_sum)};
}

const std::vector<Particle>& ParticleFilter::get_particles() const {
    return particles_;
}
