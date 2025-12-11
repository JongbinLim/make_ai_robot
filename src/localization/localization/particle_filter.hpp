#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <cstring>

// Eigen은 필수 의존성이 아니도록 평범한 구조체 사용
struct Particle {
    float x;
    float y;
    float yaw;
    float weight;
};

class ParticleFilter {
public:
    ParticleFilter(int min_particles = 500, int max_particles = 2000, 
                   float init_noise_x = 0.1f, float init_noise_y = 0.1f, float init_noise_yaw = 0.1f);
    ~ParticleFilter();

    // 초기화
    void initialize(float x, float y, float yaw);
    
    // 맵 설정 (Occupancy Grid -> Likelihood Field 변환)
    void set_map(const std::vector<int8_t>& map_data, int width, int height, float resolution, float origin_x, float origin_y);

    // 모션 모델 (Prediction)
    void predict(float dx, float dy, float dyaw);

    // 센서 모델 (Correction)
    void update(const std::vector<float>& scan_ranges, float angle_min, float angle_inc, const float sensor_offset[2]);

    // 리샘플링 (KLD + Low Variance)
    void resample();

    // 추정 위치 반환 [x, y, yaw]
    std::vector<float> get_estimated_pose();
    
    // 파티클 데이터 접근 (시각화용)
    const std::vector<Particle>& get_particles() const;

private:
    // 파라미터
    int min_particles_;
    int max_particles_;
    float init_noise_[3];
    float motion_alphas_[6]; // [a1, a2, a3, a4, a5, a6]
    float min_motion_noise_[3];
    
    // 파티클 저장소
    std::vector<Particle> particles_;
    
    // 맵 데이터 (Flattened)
    std::vector<float> log_likelihood_map_;
    std::vector<float> dist_map_; // 거리(m)
    std::vector<std::pair<int, int>> free_space_indices_; // 납치 복구용 빈 공간 좌표
    
    int map_width_;
    int map_height_;
    float map_resolution_;
    float map_origin_x_;
    float map_origin_y_;
    
    // 센서 모델 파라미터
    float sensor_sigma_;
    float sensor_model_factor_;
    float dist_threshold_;
    
    // KLD 파라미터
    float kld_err_;
    float kld_z_;

    // 난수 생성기
    std::mt19937 gen_;

    // 내부 유틸리티
    float normalize_angle(float angle);
    void recover_from_kidnapping();
    
    // 삼각함수 캐싱
    std::vector<float> cos_cache_;
    std::vector<float> sin_cache_;
    int cached_num_scans_ = -1;
    
    void update_trig_cache(int n_scans, float angle_min, float angle_inc);
};

#endif // PARTICLE_FILTER_HPP
