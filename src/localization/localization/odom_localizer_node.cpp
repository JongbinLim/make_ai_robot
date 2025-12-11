#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>

#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std::chrono_literals;

// --- 유틸리티 함수 ---
inline float normalize_angle(float angle) {
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

// --- UKF 클래스 (Eigen 기반 구현) ---
class RobotUKF {
public:
    RobotUKF(float dt) : dt_(dt) {
        x_.setZero();
        P_.setIdentity();
        P_ *= 0.1;
        Q_.setIdentity();
        Q_.diagonal() << 0.001, 0.001, 0.001, 0.01, 0.05;
        R_icp_.setIdentity();
        R_icp_.diagonal() << 0.05, 0.05, 0.02;
        R_imu_.setIdentity();
        R_imu_ << 0.02;
        alpha_ = 0.1;
        beta_ = 2.0;
        kappa_ = 0.0;
        lambda_ = alpha_ * alpha_ * (5 + kappa_) - 5;
        wm_.resize(11);
        wc_.resize(11);
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
        x_.setZero();
        for (int i = 0; i < 11; ++i) {
            x_ += wm_[i] * sigmas_pred.col(i);
        }
        
        x_(2) = std::atan2(
            (sigmas_pred.row(2).array().sin() * Eigen::Map<Eigen::ArrayXd>(wm_.data(), 11).transpose()).sum(),
            (sigmas_pred.row(2).array().cos() * Eigen::Map<Eigen::ArrayXd>(wm_.data(), 11).transpose()).sum()
        );

        P_.setZero();
        for (int i = 0; i < 11; ++i) {
            Eigen::VectorXd diff = sigmas_pred.col(i) - x_;
            diff(2) = normalize_angle(diff(2));
            P_ += wc_[i] * (diff * diff.transpose());
        }
        P_ += Q_ * (dt / dt_);
    }
    void update_icp(const Eigen::Vector3d& z, double motion_factor) {
        Eigen::MatrixXd sigmas = generate_sigma_points(x_, P_);
        Eigen::MatrixXd Z_sigmas(3, 11);
        for(int i=0; i<11; ++i) {
            Z_sigmas(0, i) = sigmas(0, i);
            Z_sigmas(1, i) = sigmas(1, i);
            Z_sigmas(2, i) = sigmas(2, i);
        }
        Eigen::Vector3d z_pred = Eigen::Vector3d::Zero();
        for(int i=0; i<11; ++i) {
            z_pred += wm_[i] * Z_sigmas.col(i);
        }
        z_pred(2) = std::atan2(
            (Z_sigmas.row(2).array().sin() * Eigen::Map<Eigen::ArrayXd>(wm_.data(), 11).transpose()).sum(),
            (Z_sigmas.row(2).array().cos() * Eigen::Map<Eigen::ArrayXd>(wm_.data(), 11).transpose()).sum()
        );
        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(5, 3);
        for(int i=0; i<11; ++i) {
            Eigen::Vector3d z_diff = Z_sigmas.col(i) - z_pred;
            z_diff(2) = normalize_angle(z_diff(2));
            S += wc_[i] * (z_diff * z_diff.transpose());
            Eigen::VectorXd x_diff = sigmas.col(i) - x_;
            x_diff(2) = normalize_angle(x_diff(2));
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
         H(4) = 1.0;
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
        double cmd_omega = u(1);
        double alpha_v = 0.1;
        double alpha_w = 0.0;
        double next_v = v + alpha_v * (cmd_v - v);
        double next_omega = omega + alpha_w * (cmd_omega - omega);
        double next_x, next_y;
        if (std::abs(omega) > 1e-5) {
            double v_w = v / omega;
            double sin_t = std::sin(theta);
            double cos_t = std::cos(theta);
            double sin_t_n = std::sin(theta + omega * dt);
            double cos_t_n = std::cos(theta + omega * dt);
            next_x = state(0) + v_w * (sin_t_n - sin_t);
            next_y = state(1) + v_w * (-cos_t_n + cos_t);
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
        kf_dist_th_ = this->declare_parameter("keyframe_dist", 0.1);
        kf_angle_th_ = this->declare_parameter("keyframe_angle", 0.1);

        auto qos = rclcpp::SensorDataQoS();
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", qos, std::bind(&OdomLocalizerNode::scan_callback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_plugin/out", qos, std::bind(&OdomLocalizerNode::imu_callback, this, std::placeholders::_1));
        
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&OdomLocalizerNode::cmd_vel_callback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        icp_.setMaximumIterations(30);
        icp_.setTransformationEpsilon(1e-8);
        icp_.setMaxCorrespondenceDistance(0.5); 
        icp_.setEuclideanFitnessEpsilon(1e-5);
        
        // --- [수정된 부분] ---
        // 오류 해결: last_cmd_time_을 노드의 현재 시간(ROS Time)으로 초기화
        // 이렇게 해야 imu_callback에서 (this->now() - last_cmd_time_) 연산 시 시간 타입 불일치 에러가 발생하지 않음
        last_cmd_time_ = this->now();
        // -------------------

        RCLCPP_INFO(this->get_logger(), "Odom Localizer Node (C++) Started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_cmd_ << msg->linear.x, msg->angular.z;
        last_cmd_time_ = this->now();
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        double current_time = rclcpp::Time(msg->header.stamp).seconds();
        double omega = msg->angular_velocity.z;

        Eigen::Vector2d u = current_cmd_;
        
        // 여기서 시간 소스가 다르면 에러가 발생했었음 -> 생성자 수정으로 해결됨
        if ((this->now() - last_cmd_time_).seconds() > 0.5) {
            u.setZero();
        }

        if (last_imu_time_ < 0) {
            last_imu_time_ = current_time;
            return;
        }

        double dt = current_time - last_imu_time_;
        if (dt <= 0) return;

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

    double get_interpolated_omega(double t) {
        if (imu_history_.empty()) return 0.0;
        if (t <= imu_history_.front().first) return imu_history_.front().second;
        if (t >= imu_history_.back().first) return imu_history_.back().second;

        for (size_t i = 0; i < imu_history_.size() - 1; ++i) {
            if (imu_history_[i].first <= t && t < imu_history_[i+1].first) {
                double r = (t - imu_history_[i].first) / (imu_history_[i+1].first - imu_history_[i].first);
                return imu_history_[i].second + r * (imu_history_[i+1].second - imu_history_[i].second);
            }
        }
        return 0.0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr deskew_scan(const sensor_msgs::msg::LaserScan::SharedPtr& msg, double v, double omega) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->reserve(msg->ranges.size());

        double angle_min = msg->angle_min;
        double angle_inc = msg->angle_increment;
        double time_inc = msg->time_increment;
        if (time_inc < 1e-9 && !msg->ranges.empty()) time_inc = msg->scan_time / msg->ranges.size();

        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (r < msg->range_min || r > msg->range_max || std::isnan(r) || std::isinf(r)) continue;

            double dt = i * time_inc;
            double delta_theta = omega * dt;
            double delta_x, delta_y;

            if (std::abs(omega) > 1e-4) {
                double radius = v / omega;
                delta_x = radius * std::sin(delta_theta);
                delta_y = radius * (1.0 - std::cos(delta_theta));
            } else {
                delta_x = v * dt;
                delta_y = 0.0;
            }

            double theta_point = angle_min + i * angle_inc;
            double corrected_theta = theta_point + delta_theta;

            pcl::PointXYZ p; 
            p.x = r * std::cos(corrected_theta) + delta_x;
            p.y = r * std::sin(corrected_theta) + delta_y;
            p.z = 0.0; 
            cloud->push_back(p);
        }
        return cloud;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        double scan_time = rclcpp::Time(msg->header.stamp).seconds();

        Eigen::VectorXd state = ukf_.get_state();
        double v = state(3);
        
        double omega_interp = get_interpolated_omega(scan_time);
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud = deskew_scan(msg, v, omega_interp);

        if (current_cloud->size() < 30) return;

        if (!keyframe_cloud_) {
            keyframe_cloud_ = current_cloud;
            last_keyframe_time_ = scan_time;
            last_keyframe_pose_ = state.head<3>();
            return;
        }

        double kf_yaw = last_keyframe_pose_(2);
        
        double dx = state(0) - last_keyframe_pose_(0);
        double dy = state(1) - last_keyframe_pose_(1);
        double dth = normalize_angle(state(2) - kf_yaw);

        double c_k = std::cos(kf_yaw), s_k = std::sin(kf_yaw);
        double local_x = c_k * dx + s_k * dy;
        double local_y = -s_k * dx + c_k * dy;

        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        guess(0, 3) = local_x;
        guess(1, 3) = local_y;
        guess.block<2,2>(0,0) << std::cos(dth), -std::sin(dth), std::sin(dth), std::cos(dth);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        icp_.setInputSource(current_cloud);
        icp_.setInputTarget(keyframe_cloud_);
        icp_.align(*aligned, guess);

        if (!icp_.hasConverged() || icp_.getFitnessScore() > 1.0) {
            if (dx*dx + dy*dy > kf_dist_th_*kf_dist_th_*4.0) {
                 keyframe_cloud_ = current_cloud;
                 last_keyframe_time_ = scan_time;
                 last_keyframe_pose_ = state.head<3>();
                 RCLCPP_WARN(this->get_logger(), "ICP Failed but moved far. Reset Keyframe.");
            }
            return;
        }

        Eigen::Matrix4f T_rel = icp_.getFinalTransformation();
        double dx_rel = T_rel(0, 3);
        double dy_rel = T_rel(1, 3);
        double dth_rel = std::atan2(T_rel(1, 0), T_rel(0, 0));

        double gx = last_keyframe_pose_(0) + c_k * dx_rel - s_k * dy_rel;
        double gy = last_keyframe_pose_(1) + s_k * dx_rel + c_k * dy_rel;
        double gth = normalize_angle(kf_yaw + dth_rel);

        Eigen::Vector3d measurement(gx, gy, gth);

        double lag_penalty = 1.0; 
        double motion_factor = 1.0 + 3.0 * std::abs(omega_interp) + 2.0 * std::abs(v) * lag_penalty;
        ukf_.update_icp(measurement, motion_factor);

        state = ukf_.get_state();
        double dist_sq = std::pow(state(0) - last_keyframe_pose_(0), 2) + std::pow(state(1) - last_keyframe_pose_(1), 2);
        double ang_diff = std::abs(normalize_angle(state(2) - last_keyframe_pose_(2)));

        if (dist_sq > kf_dist_th_ * kf_dist_th_ || ang_diff > kf_angle_th_) {
            keyframe_cloud_ = current_cloud;
            last_keyframe_time_ = scan_time;
            last_keyframe_pose_ = state.head<3>();
        }
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
    double kf_dist_th_, kf_angle_th_;

    Eigen::Vector2d current_cmd_ = Eigen::Vector2d::Zero();
    rclcpp::Time last_cmd_time_; // 생성자에서 초기화됨
    double last_imu_time_ = -1.0;
    
    std::deque<std::pair<double, double>> imu_history_; 

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe_cloud_;
    double last_keyframe_time_;
    Eigen::Vector3d last_keyframe_pose_;
    
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomLocalizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
