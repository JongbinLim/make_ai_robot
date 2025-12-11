#include "particle_filter.hpp"
#include <queue>
#include <omp.h> // 병렬 처리를 위한 OpenMP 헤더

// 상수 정의
constexpr float PI = 3.14159265359f;
constexpr float TWO_PI = 2.0f * PI;

ParticleFilter::ParticleFilter(int min_particles, int max_particles, 
                               float init_noise_x, float init_noise_y, float init_noise_yaw)
    : min_particles_(min_particles), max_particles_(max_particles),
      map_width_(0), map_height_(0),
      sensor_sigma_(0.3f), kld_err_(0.015f), kld_z_(2.326f) {
    
    init_noise_[0] = init_noise_x;
    init_noise_[1] = init_noise_y;
    init_noise_[2] = init_noise_yaw;

    // Motion Alphas (Python 코드와 동일)
    float alphas[] = {0.09f, 0.09f, 0.06f, 0.09f, 0.06f, 0.09f};
    std::memcpy(motion_alphas_, alphas, sizeof(alphas));
    
    min_motion_noise_[0] = 0.05f;
    min_motion_noise_[1] = 0.05f;
    min_motion_noise_[2] = 0.05f;

    sensor_model_factor_ = -0.5f / (sensor_sigma_ * sensor_sigma_);
    dist_threshold_ = sensor_sigma_ * 3.0f;

    std::random_device rd;
    gen_ = std::mt19937(rd());
}

ParticleFilter::~ParticleFilter() {}

float ParticleFilter::normalize_angle(float angle) {
    angle = fmod(angle + PI, TWO_PI);
    if (angle < 0) angle += TWO_PI;
    return angle - PI;
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

// Occupancy Grid를 거리 맵(Distance Transform)으로 변환 (BFS 사용)
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

    // 1. 초기화 및 Obstacle Queue 삽입
    std::queue<int> q;
    std::vector<bool> visited(size, false);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int8_t val = map_data[idx];

            if (val >= 50) { // 장애물
                dist_map_[idx] = 0.0f;
                q.push(idx);
                visited[idx] = true;
            } else if (val >= 0) { // 빈 공간
                free_space_indices_.push_back({x, y});
            }
        }
    }

    // 2. BFS로 Distance Transform (Manhattan/Euclidean 근사)
    // 정확한 EDT보다는 BFS Brushfire 알고리즘이 구현이 간단하고 충분히 빠름
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        int curr_idx = q.front();
        q.pop();

        int cx = curr_idx % width;
        int cy = curr_idx / width;

        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                if (!visited[n_idx]) {
                    // 거리 갱신 (인접 픽셀 거리 + 1 * 해상도)
                    // 여기서는 픽셀 단위 BFS 후 해상도를 곱하는 방식을 사용 (Manhattan distance)
                    // 더 부드러운 분포를 위해 Euclidean을 쓰려면 복잡해지므로 근사 사용
                    dist_map_[n_idx] = dist_map_[curr_idx] + resolution;
                    visited[n_idx] = true;
                    q.push(n_idx);
                }
            }
        }
    }

    // 3. Likelihood Field 생성
    float min_log_prob = -10.0f;
    for (int i = 0; i < size; ++i) {
        if (dist_map_[i] == std::numeric_limits<float>::max()) {
            dist_map_[i] = 100.0f; // Unreachable
        }
        
        // Gaussian Model
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

    // OpenMP 병렬화
    #pragma omp parallel
    {
        // 스레드별 난수 생성기
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num());
        std::normal_distribution<float> noise_x(0.0f, sigma_x);
        std::normal_distribution<float> noise_y(0.0f, sigma_y);
        std::normal_distribution<float> noise_yaw(0.0f, sigma_yaw);

        #pragma omp for
        for (size_t i = 0; i < particles_.size(); ++i) {
            float nx = noise_x(local_gen);
            float ny = noise_y(local_gen);
            float nyaw = noise_yaw(local_gen);

            float noisy_dx = dx + nx;
            float noisy_dy = dy + ny;
            float noisy_dyaw = dyaw + nyaw;

            float c = std::cos(particles_[i].yaw);
            float s = std::sin(particles_[i].yaw);

            particles_[i].x += (noisy_dx * c - noisy_dy * s);
            particles_[i].y += (noisy_dx * s + noisy_dy * c);
            particles_[i].yaw = normalize_angle(particles_[i].yaw + noisy_dyaw);
        }
    }
}

void ParticleFilter::update_trig_cache(int n_scans, float angle_min, float angle_inc) {
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
    
    int step = 4; // Downsampling
    int n_scans = scan_ranges.size();
    
    // 캐시 업데이트
    // step을 고려하여 실제 루프 돌릴 인덱스용 삼각함수만 필요하지만, 
    // 여기서는 전체 캐시 후 인덱싱 사용
    if (n_scans != cached_num_scans_) {
        update_trig_cache(n_scans, angle_min, angle_inc);
    }

    float inv_res = 1.0f / map_resolution_;
    float min_log_prob = -10.0f;
    float penalty_for_dynamic = -1.5f;

    // Log weight 저장을 위한 임시 벡터
    std::vector<float> log_weights(particles_.size());
    float max_log_w = -1e15f;

    #pragma omp parallel for reduction(max: max_log_w)
    for (size_t i = 0; i < particles_.size(); ++i) {
        float px = particles_[i].x;
        float py = particles_[i].y;
        float pyaw = particles_[i].yaw;
        float c = std::cos(pyaw);
        float s = std::sin(pyaw);

        float total_log_score = 0.0f;

        for (int j = 0; j < n_scans; j += step) {
            float r = scan_ranges[j];
            if (r < 0.01f || r > 20.0f) continue;

            // 로봇 프레임 좌표
            float r_cos = r * cos_cache_[j];
            float r_sin = r * sin_cache_[j];

            float lx = r_cos + sensor_offset[0];
            float ly = r_sin + sensor_offset[1];

            // 월드 좌표 변환
            float wx = px + (c * lx - s * ly);
            float wy = py + (s * lx + c * ly);

            int mx = static_cast<int>((wx - map_origin_x_) * inv_res);
            int my = static_cast<int>((wy - map_origin_y_) * inv_res);

            float score = min_log_prob;

            if (mx >= 0 && mx < map_width_ && my >= 0 && my < map_height_) {
                int idx = my * map_width_ + mx;
                score = log_likelihood_map_[idx];
                float dist = dist_map_[idx];

                // Robust Likelihood (동적 장애물 처리)
                if (dist > dist_threshold_ && score < penalty_for_dynamic) {
                    score = penalty_for_dynamic;
                }
            }
            total_log_score += score;
        }
        log_weights[i] = total_log_score;
        if (total_log_score > max_log_w) max_log_w = total_log_score;
    }

    // Weight Normalization (Log-Sum-Exp Trick)
    float sum_w = 0.0f;
    for (size_t i = 0; i < particles_.size(); ++i) {
        float w = std::exp(log_weights[i] - max_log_w);
        particles_[i].weight = w;
        sum_w += w;
    }

    // Kidnapping Check
    if (sum_w < 1e-9f || std::isnan(sum_w)) {
        recover_from_kidnapping();
    } else {
        float sum_sq_w = 0.0f;
        for (auto& p : particles_) {
            p.weight /= sum_w;
            sum_sq_w += p.weight * p.weight;
        }

        // Resample Check (N_eff)
        if (1.0f / sum_sq_w < particles_.size() / 1.5f) {
            resample();
        }
    }
}

void ParticleFilter::resample() {
    // 1. KLD Sampling Number Calculation
    // Grid Hash를 이용한 Bin Count
    float xy_res = 0.1f; 
    float yaw_res = 2.0f * PI / 180.0f; // 2 degrees

    // 입자들을 Grid Index 키로 변환하여 유니크 개수 산출
    std::vector<long> bins;
    bins.reserve(particles_.size());
    for(const auto& p : particles_) {
        long kx = static_cast<long>(p.x / xy_res);
        long ky = static_cast<long>(p.y / xy_res);
        long kyaw = static_cast<long>(p.yaw / yaw_res);
        // Simple Hash Key
        bins.push_back(kx + ky * 100000 + kyaw * 10000000000L);
    }
    std::sort(bins.begin(), bins.end());
    int k = std::unique(bins.begin(), bins.end()) - bins.begin(); // 유니크 Bin 개수

    int new_n = min_particles_;
    if (k > 1) {
        float term1 = 1.0f - 2.0f / (9.0f * (k - 1));
        float term2 = std::sqrt(2.0f / (9.0f * (k - 1))) * kld_z_;
        float term3 = term1 + term2;
        new_n = static_cast<int>((k - 1) / (2.0f * kld_err_) * (term3 * term3 * term3));
    }
    new_n = std::max(min_particles_, std::min(new_n, max_particles_));

    // 2. Low Variance Resampling
    std::vector<Particle> new_particles;
    new_particles.reserve(new_n);

    // 가중치가 가장 높은 파티클 보존 (Elitism)
    auto best_it = std::max_element(particles_.begin(), particles_.end(), 
        [](const Particle& a, const Particle& b){ return a.weight < b.weight; });
    new_particles.push_back(*best_it);

    float step = 1.0f / new_n;
    std::uniform_real_distribution<float> dist(0.0f, step);
    float r = dist(gen_);
    
    float c = particles_[0].weight;
    int idx = 0;

    for (int i = 1; i < new_n; ++i) {
        float u = r + (float)i * step;
        while (u > c && idx < (int)particles_.size() - 1) {
            idx++;
            c += particles_[idx].weight;
        }
        new_particles.push_back(particles_[idx]);
        new_particles.back().weight = 1.0f / new_n; // Reset weights
    }

    particles_ = new_particles;
}

void ParticleFilter::recover_from_kidnapping() {
    if (free_space_indices_.empty()) return;

    std::cout << "[PF] Kidnap detected. Injecting random particles." << std::endl;
    
    int n_keep = static_cast<int>(particles_.size() * 0.3);
    int n_random = particles_.size() - n_keep;

    // 상위 파티클 정렬
    std::sort(particles_.begin(), particles_.end(), 
              [](const Particle& a, const Particle& b) { return a.weight > b.weight; });
    
    particles_.resize(n_keep + n_random);

    std::uniform_int_distribution<int> idx_dist(0, free_space_indices_.size() - 1);
    std::uniform_real_distribution<float> offset_dist(0.0f, map_resolution_);
    std::uniform_real_distribution<float> yaw_dist(-PI, PI);

    for (int i = n_keep; i < particles_.size(); ++i) {
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

    // 상위 20% 가중 평균
    std::vector<int> indices(particles_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j){
        return particles_[i].weight > particles_[j].weight;
    });

    int n_top = std::max(5, (int)(particles_.size() * 0.2));
    
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
