#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <unordered_map>
#include <omp.h>

using namespace std::chrono_literals;

// --- 유틸리티 함수 ---
inline double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// --- Voxel Hashing & Local Map (KISS-ICP Style) ---
struct VoxelHash {
    size_t operator()(const Eigen::Vector2i& k) const {
        return (size_t)((k.x() * 73856093) ^ (k.y() * 19349663));
    }
};

struct Voxel {
    Eigen::Vector2d sum_pos = Eigen::Vector2d::Zero();
    Eigen::Matrix2d sum_cov = Eigen::Matrix2d::Zero();
    int count = 0;

    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    bool is_ready = false;

    void add(const Eigen::Vector2d& p) {
        sum_pos += p;
        sum_cov += p * p.transpose();
        count++;
    }

    void compute_model() {
        if (count < 5) {
            is_ready = false;
            return;
        }
        mean = sum_pos / count;
        Eigen::Matrix2d cov = (sum_cov / count) - (mean * mean.transpose());

        double trace = cov.trace();
        double det = cov.determinant();
        double lambda_min = (trace - std::sqrt(std::max(0.0, trace*trace - 4*det))) / 2.0;

        Eigen::Vector2d ev;
        if (std::abs(cov(0,1)) < 1e-9) {
            ev = (cov(0,0) < cov(1,1)) ? Eigen::Vector2d(1,0) : Eigen::Vector2d(0,1);
        } else {
            ev = Eigen::Vector2d(cov(0,1), lambda_min - cov(0,0)).normalized();
        }
        normal = ev;
        is_ready = true;
    }
};

struct VoxelGridMap {
    // 맵 크기를 고정 (예: 100x100m 영역 커버 가능한 해시 테이블)
    // 충돌 처리를 단순화하기 위해 크기를 충분히 크게 잡음 (2^16 = 65536 등)
    static constexpr size_t TABLE_SIZE = 131072; // 2^17
    static constexpr size_t MASK = TABLE_SIZE - 1;

    struct Entry {
        Eigen::Vector2i key;
        Voxel voxel;
        bool active = false;
    };

    std::vector<Entry> table;
    double voxel_size;

    VoxelGridMap() : voxel_size(0.2) {
        table.resize(TABLE_SIZE);
    }

    void clear() {
        std::fill(table.begin(), table.end(), Entry{});
    }

    // 간단한 해시 함수
    size_t hash(const Eigen::Vector2i& k) const {
        return ((k.x() * 73856093) ^ (k.y() * 19349663)) & MASK;
    }

    void add_cloud(const std::vector<Eigen::Vector2d>& points, const Eigen::Vector3d& pose) {
        double c = std::cos(pose(2)), s = std::sin(pose(2));
        Eigen::Matrix2d R; R << c, -s, s, c;
        Eigen::Vector2d t = pose.head<2>();

        for (const auto& p_local : points) {
            Eigen::Vector2d p_world = R * p_local + t;
            Eigen::Vector2i key;
            key.x() = static_cast<int>(std::floor(p_world.x() / voxel_size));
            key.y() = static_cast<int>(std::floor(p_world.y() / voxel_size));

            size_t idx = hash(key);

            // Linear Probing (최대 5칸까지만 검색하고 포기 - 속도 우선)
            for(int i=0; i<5; ++i) {
                size_t curr = (idx + i) & MASK;
                if (!table[curr].active) {
                    table[curr].active = true;
                    table[curr].key = key;
                    table[curr].voxel = Voxel(); // reset
                    table[curr].voxel.add(p_world);
                    break;
                } else if (table[curr].key == key) {
                    table[curr].voxel.add(p_world);
                    break;
                }
            }
        }
    }

    void update_voxels() {
        #pragma omp parallel for
        for(size_t i=0; i<TABLE_SIZE; ++i) {
            if(table[i].active) {
                table[i].voxel.compute_model();
            }
        }
    }

    // 9-Neighbor Search 최적화
    bool get_closest_line(const Eigen::Vector2d& pt_world, double max_dist,
                          Eigen::Vector2d& out_mean, Eigen::Vector2d& out_normal) const {
        Eigen::Vector2i center_key;
        center_key.x() = static_cast<int>(std::floor(pt_world.x() / voxel_size));
        center_key.y() = static_cast<int>(std::floor(pt_world.y() / voxel_size));

        double min_dist_sq = max_dist * max_dist;
        const Voxel* best_voxel = nullptr;

        // Loop Unrolling 및 해시 룩업 최소화
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                Eigen::Vector2i query = center_key + Eigen::Vector2i(dx, dy);
                size_t idx = hash(query);

                // Linear Probing 탐색
                for(int i=0; i<5; ++i) {
                    size_t curr = (idx + i) & MASK;
                    if(!table[curr].active) break; // 빈칸이면 종료 (Cluster assumption)

                    if(table[curr].key == query && table[curr].voxel.is_ready) {
                         double dist_sq = (table[curr].voxel.mean - pt_world).squaredNorm();
                         if (dist_sq < min_dist_sq) {
                             min_dist_sq = dist_sq;
                             best_voxel = &(table[curr].voxel);
                         }
                         break; // 키를 찾았으면 다음 dx/dy로
                    }
                }
            }
        }

        if (best_voxel) {
            out_mean = best_voxel->mean;
            out_normal = best_voxel->normal;
            return true;
        }
        return false;
    }

    // remove_far_voxels 최적화: 전체 순회하며 거리 체크
    void remove_far_voxels(const Eigen::Vector2d& center, double radius) {
        double r_sq = radius * radius;
        #pragma omp parallel for
        for(size_t i=0; i<TABLE_SIZE; ++i) {
            if(table[i].active) {
                Eigen::Vector2d dist = table[i].voxel.mean - center;
                if (dist.squaredNorm() > r_sq) {
                    table[i].active = false; // Soft delete
                }
            }
        }
    }
};

// --- KISS-ICP Solver (Gauss-Newton) ---
struct KISSICPSolver {
    int max_iterations;
    double tolerance;
    double adaptive_threshold_initial;
    double adaptive_threshold_min;

    KISSICPSolver() : max_iterations(20), tolerance(1e-4),
                      adaptive_threshold_initial(1.0), adaptive_threshold_min(0.2) {}

    Eigen::Vector3d align(const std::vector<Eigen::Vector2d>& src,
                          const VoxelGridMap& map,
                          Eigen::Vector3d initial_pose,
                          double& final_rmse) {

        Eigen::Vector3d current_pose = initial_pose;

        for (int iter = 0; iter < max_iterations; ++iter) {
            double threshold = adaptive_threshold_initial -
                (adaptive_threshold_initial - adaptive_threshold_min) * ((double)iter / max_iterations);

            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            Eigen::Vector3d b = Eigen::Vector3d::Zero();
            double total_residual = 0.0;
            int valid_points = 0;

            double c = std::cos(current_pose(2));
            double s = std::sin(current_pose(2));
            Eigen::Matrix2d R; R << c, -s, s, c;
            Eigen::Vector2d t = current_pose.head<2>();

            // OpenMP를 사용한 병렬 Jacobian 계산
            // reduction은 복잡하므로 각 스레드별 로컬 변수에 누적 후 합산
            #pragma omp parallel
            {
                Eigen::Matrix3d H_local = Eigen::Matrix3d::Zero();
                Eigen::Vector3d b_local = Eigen::Vector3d::Zero();
                double res_local = 0.0;
                int valid_local = 0;

                #pragma omp for nowait
                for (size_t i = 0; i < src.size(); ++i) {
                    Eigen::Vector2d p_world = R * src[i] + t;
                    Eigen::Vector2d map_mean, map_normal;

                    if (map.get_closest_line(p_world, threshold, map_mean, map_normal)) {
                        Eigen::Vector2d diff = p_world - map_mean;
                        double residual = map_normal.dot(diff);

                        if (std::abs(residual) < threshold) {
                            double j_ang = map_normal.x() * (-s * src[i].x() - c * src[i].y()) +
                                           map_normal.y() * ( c * src[i].x() - s * src[i].y());
                            Eigen::Vector3d J(map_normal.x(), map_normal.y(), j_ang);

                            H_local += J * J.transpose();
                            b_local += -residual * J;
                            res_local += residual * residual;
                            valid_local++;
                        }
                    }
                }

                #pragma omp critical
                {
                    H += H_local;
                    b += b_local;
                    total_residual += res_local;
                    valid_points += valid_local;
                }
            }

            if (valid_points < 10) break;

            Eigen::Vector3d dx = H.ldlt().solve(b);
            if (std::isnan(dx(0))) break;

            current_pose += dx;
            current_pose(2) = normalize_angle(current_pose(2));

            final_rmse = std::sqrt(total_residual / valid_points);
            if (dx.norm() < tolerance) break;
        }
        return current_pose;
    }
};

// --- UKF Implementation (복원 및 최적화) ---
class RobotUKF {
public:
    RobotUKF(float dt) : dt_(dt) {
        x_.setZero();
        P_.setIdentity(); P_ *= 0.1;
        Q_.setIdentity(); Q_.diagonal() << 0.001, 0.001, 0.001, 0.01, 0.05;
        R_icp_.setIdentity(); R_icp_.diagonal() << 0.05, 0.05, 0.02;
        R_imu_.setIdentity(); R_imu_ << 0.02;

        alpha_ = 0.1; beta_ = 2.0; kappa_ = 0.0;
        lambda_ = alpha_ * alpha_ * (5 + kappa_) - 5;
        wm_.resize(11); wc_.resize(11);
        wm_[0] = lambda_ / (5 + lambda_);
        wc_[0] = wm_[0] + (1 - alpha_ * alpha_ + beta_);
        for (int i = 1; i < 11; ++i) {
            wm_[i] = 1.0 / (2 * (5 + lambda_));
            wc_[i] = wm_[i];
        }
    }

    Eigen::VectorXd get_state() { return x_; }

    void predict(float dt, const Eigen::Vector2d& u) {
        Eigen::MatrixXd sigmas = generate_sigma_points(x_, P_);
        Eigen::MatrixXd sigmas_pred(5, 11);
        for (int i = 0; i < 11; ++i) {
            sigmas_pred.col(i) = motion_model(sigmas.col(i), dt, u);
        }

        // Mean Prediction
        x_.setZero();
        for (int i = 0; i < 11; ++i) x_ += wm_[i] * sigmas_pred.col(i);

        // Angle Averaging
        double sin_sum = 0.0, cos_sum = 0.0;
        for (int i=0; i<11; ++i) {
             sin_sum += wm_[i] * std::sin(sigmas_pred(2,i));
             cos_sum += wm_[i] * std::cos(sigmas_pred(2,i));
        }
        x_(2) = std::atan2(sin_sum, cos_sum);

        // Covariance Prediction
        P_.setZero();
        for (int i = 0; i < 11; ++i) {
            Eigen::VectorXd diff = sigmas_pred.col(i) - x_;
            diff(2) = normalize_angle(diff(2));
            P_ += wc_[i] * (diff * diff.transpose());
        }
        P_ += Q_ * (dt / dt_);
    }

    // ICP 결과를 Measurement로 사용 (z: [x, y, theta])
    void update_icp(const Eigen::Vector3d& z, double motion_factor) {
        Eigen::MatrixXd sigmas = generate_sigma_points(x_, P_);
        Eigen::MatrixXd Z_sigmas(3, 11);
        for(int i=0; i<11; ++i) Z_sigmas.col(i) = sigmas.col(i).head<3>();

        Eigen::Vector3d z_pred = Eigen::Vector3d::Zero();
        for(int i=0; i<11; ++i) z_pred += wm_[i] * Z_sigmas.col(i);

        double sin_sum = 0.0, cos_sum = 0.0;
        for(int i=0; i<11; ++i) {
            sin_sum += wm_[i] * std::sin(Z_sigmas(2,i));
            cos_sum += wm_[i] * std::cos(Z_sigmas(2,i));
        }
        z_pred(2) = std::atan2(sin_sum, cos_sum);

        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(5, 3);

        for(int i=0; i<11; ++i) {
            Eigen::Vector3d z_diff = Z_sigmas.col(i) - z_pred;
            z_diff(2) = normalize_angle(z_diff(2));

            Eigen::VectorXd x_diff = sigmas.col(i) - x_;
            x_diff(2) = normalize_angle(x_diff(2));

            S += wc_[i] * (z_diff * z_diff.transpose());
            Pxz += wc_[i] * (x_diff * z_diff.transpose());
        }

        Eigen::Matrix3d R = R_icp_ * motion_factor;
        S += R;

        Eigen::MatrixXd K = Pxz * S.inverse();
        Eigen::Vector3d y = z - z_pred;
        y(2) = normalize_angle(y(2));

        x_ += K * y;
        x_(2) = normalize_angle(x_(2));
        P_ -= K * S * K.transpose();
    }

    void update_imu(double omega) {
         Eigen::RowVectorXd H = Eigen::RowVectorXd::Zero(5);
         H(4) = 1.0; // State index 4 is omega

         double z_pred = x_(4);
         double y = omega - z_pred;
         double S = (H * P_ * H.transpose())(0, 0) + R_imu_(0,0);

         Eigen::VectorXd K = P_ * H.transpose() / S;
         x_ += K * y;
         P_ -= K * S * K.transpose();
    }

private:
    Eigen::MatrixXd generate_sigma_points(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        Eigen::MatrixXd sigmas(5, 11);
        Eigen::LLT<Eigen::MatrixXd> llt(P + Eigen::MatrixXd::Identity(5,5)*1e-9);
        Eigen::MatrixXd L = llt.matrixL();
        double scale = std::sqrt(5 + lambda_);
        sigmas.col(0) = x;
        for (int i = 0; i < 5; ++i) {
            sigmas.col(i + 1) = x + scale * L.col(i);
            sigmas.col(i + 6) = x - scale * L.col(i);
        }
        for(int i=0; i<11; ++i) sigmas(2, i) = normalize_angle(sigmas(2, i));
        return sigmas;
    }

    Eigen::VectorXd motion_model(Eigen::VectorXd state, float dt, const Eigen::Vector2d& u) {
        double theta = state(2);
        double v = state(3);
        double omega = state(4);
        double cmd_v = u(0);
        // double cmd_omega = u(1); // Not used directly in simple model, implicit in omega

        // Simple velocity decay/response model
        double next_v = v + 0.1 * (cmd_v - v);
        double next_omega = omega; // Assume constant angular velocity for short dt

        double next_x, next_y;
        if (std::abs(omega) > 1e-5) {
            double v_w = v / omega;
            next_x = state(0) + v_w * (std::sin(theta + omega * dt) - std::sin(theta));
            next_y = state(1) + v_w * (-std::cos(theta + omega * dt) + std::cos(theta));
        } else {
            next_x = state(0) + v * std::cos(theta + omega * dt * 0.5) * dt;
            next_y = state(1) + v * std::sin(theta + omega * dt * 0.5) * dt;
        }
        double next_theta = normalize_angle(theta + omega * dt);

        Eigen::VectorXd next_state(5);
        next_state << next_x, next_y, next_theta, next_v, next_omega;
        return next_state;
    }

    Eigen::VectorXd x_ = Eigen::VectorXd::Zero(5);
    Eigen::MatrixXd P_ = Eigen::MatrixXd::Identity(5, 5);
    Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(5, 5);
    Eigen::Matrix3d R_icp_;
    Eigen::Matrix<double, 1, 1> R_imu_;
    float dt_;
    double alpha_, beta_, kappa_, lambda_;
    std::vector<double> wm_, wc_;
};

// --- ROS Node ---
class OdomLocalizerNode : public rclcpp::Node {
public:
    OdomLocalizerNode() : Node("odom_localizer"), ukf_(0.05) {
        odom_frame_ = this->declare_parameter("odom_frame", "odom");
        base_frame_ = this->declare_parameter("base_frame", "base");

        auto qos = rclcpp::SensorDataQoS();
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", qos, std::bind(&OdomLocalizerNode::scan_callback, this, std::placeholders::_1));

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_plugin/out", qos, std::bind(&OdomLocalizerNode::imu_callback, this, std::placeholders::_1));

        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&OdomLocalizerNode::cmd_vel_callback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        local_map_.voxel_size = 0.3;

        last_cmd_time_ = this->now().seconds();

        RCLCPP_INFO(this->get_logger(), "Odom Localizer (KISS-ICP Style) Started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_cmd_ << msg->linear.x, msg->angular.z;
        last_cmd_time_ = this->now().seconds();
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        double current_time = rclcpp::Time(msg->header.stamp).seconds();
        double omega = msg->angular_velocity.z;

        Eigen::Vector2d u = current_cmd_;
        double now_sec = this->now().seconds();
        if ((now_sec - last_cmd_time_) > 0.5) {
            u.setZero(); // 명령이 끊긴지 0.5초가 지나면 정지로 간주
        }

        if (last_imu_time_ < 0) {
            last_imu_time_ = current_time;
            return;
        }

        double dt = current_time - last_imu_time_;
        if (dt <= 0) return;

        // UKF Prediction Step (Sub-stepping for stability)
        double max_step = 0.05;
        double remain = dt;
        while (remain > 1e-6) {
            double d = std::min(max_step, remain);
            ukf_.predict(d, u);
            remain -= d;
        }

        ukf_.update_imu(omega);
        publish_tf(msg->header.stamp);

        last_imu_time_ = current_time;
        imu_history_.push_back({current_time, omega});
        if (imu_history_.size() > 2000) imu_history_.pop_front();
    }

    // IMU 보간 함수 (Deskewing용)
    double get_interpolated_omega(double t) {
        if (imu_history_.empty()) return 0.0;
        if (t <= imu_history_.front().first) return imu_history_.front().second;
        if (t >= imu_history_.back().first) return imu_history_.back().second;

        auto it = std::lower_bound(imu_history_.begin(), imu_history_.end(), std::make_pair(t, -1e9),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b){
                return a.first < b.first;
            });

        if (it == imu_history_.begin()) return it->second;
        auto prev = std::prev(it);

        double dt = it->first - prev->first;
        if (dt < 1e-9) return prev->second;

        double r = (t - prev->first) / dt;
        return prev->second + r * (it->second - prev->second);
    }

    std::vector<Eigen::Vector2d> deskew_scan(const sensor_msgs::msg::LaserScan::SharedPtr& msg, double v, double omega) {
        std::vector<Eigen::Vector2d> points;
        points.resize(msg->ranges.size());

        double angle_min = msg->angle_min;
        double angle_inc = msg->angle_increment;
        double time_inc = msg->time_increment;
        if (time_inc < 1e-9) time_inc = msg->scan_time / std::max((size_t)1, msg->ranges.size());

        // Parallel Deskewing
        #pragma omp parallel for
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (r < msg->range_min || r > msg->range_max || std::isnan(r)) {
                points[i] = Eigen::Vector2d(0,0);
                continue;
            }

            double dt = i * time_inc;
            double delta_theta = omega * dt;

            // Simple Linear motion compensation
            double delta_x = v * dt;
            double delta_y = 0.0;

            // If rotating fast, use arc motion
            if (std::abs(omega) > 1e-4) {
                 double radius = v / omega;
                 delta_x = radius * std::sin(delta_theta);
                 delta_y = radius * (1.0 - std::cos(delta_theta));
            }

            double theta = angle_min + i * angle_inc + delta_theta;
            points[i] = Eigen::Vector2d(r * std::cos(theta) + delta_x, r * std::sin(theta) + delta_y);
        }

        // 유효 포인트 필터링
        std::vector<Eigen::Vector2d> valid_points;
        valid_points.reserve(points.size());
        for(const auto& p : points) {
            if(p.norm() > 0.01) valid_points.push_back(p);
        }
        return valid_points;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        double scan_time = rclcpp::Time(msg->header.stamp).seconds();

        Eigen::VectorXd state = ukf_.get_state();
        double v = state(3);

        // IMU History에서 해당 스캔 시간의 각속도 추정 (더 정확한 Deskewing을 위해)
        double omega_interp = get_interpolated_omega(scan_time);

        std::vector<Eigen::Vector2d> current_points = deskew_scan(msg, v, omega_interp);

        if (current_points.size() < 50) return;

        // 맵 초기화
        if (!map_initialized_) {
            local_map_.add_cloud(current_points, state.head<3>());
            local_map_.update_voxels();
            map_initialized_ = true;
            last_keyframes_pose_ = state.head<3>();
            return;
        }

        // KISS-ICP Align
        Eigen::Vector3d guess_pose = state.head<3>();
        double rmse = 0.0;

        Eigen::Vector3d aligned_pose = icp_solver_.align(
            current_points, local_map_, guess_pose, rmse
        );

        // UKF Update
        double motion_factor = 1.0 + 3.0 * std::abs(omega_interp) + 1.0 * std::abs(v);
        if (rmse < 0.5) {
            ukf_.update_icp(aligned_pose, motion_factor);
        } else {
            RCLCPP_WARN(this->get_logger(), "ICP Diverged (RMSE: %.2f)", rmse);
        }

        // Keyframe 관리 (Local Map Update)
        Eigen::Vector3d current_ukf_pose = ukf_.get_state().head<3>();
        double dist_moved = (current_ukf_pose.head<2>() - last_keyframes_pose_.head<2>()).norm();
        double angle_moved = std::abs(normalize_angle(current_ukf_pose(2) - last_keyframes_pose_(2)));

        if (dist_moved > 0.5 || angle_moved > 0.2) {
            local_map_.add_cloud(current_points, current_ukf_pose);
            local_map_.update_voxels();
            local_map_.remove_far_voxels(current_ukf_pose.head<2>(), 20.0);
            last_keyframes_pose_ = current_ukf_pose;
        }

        // Scan Callback에서는 TF를 쏘지 않음 (IMU Callback 주기가 더 빠르므로 거기서 담당)
        // 필요하다면 여기서도 쏠 수 있음
    }

    void publish_tf(const rclcpp::Time& timestamp) {
        Eigen::VectorXd state = ukf_.get_state();
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = timestamp;
        t.header.frame_id = odom_frame_;
        t.child_frame_id = base_frame_;
        t.transform.translation.x = state(0);
        t.transform.translation.y = state(1);
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, state(2));
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(t);
    }

    RobotUKF ukf_;
    std::mutex mutex_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string odom_frame_, base_frame_;

    Eigen::Vector2d current_cmd_ = Eigen::Vector2d::Zero();
    double last_cmd_time_;
    double last_imu_time_ = -1.0;
    std::deque<std::pair<double, double>> imu_history_;

    VoxelGridMap local_map_;
    KISSICPSolver icp_solver_;
    bool map_initialized_ = false;
    Eigen::Vector3d last_keyframes_pose_ = Eigen::Vector3d::Zero();
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<OdomLocalizerNode>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
