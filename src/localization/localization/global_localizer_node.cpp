#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "particle_filter.hpp"

using namespace std::chrono_literals;

class GlobalLocalizerNode : public rclcpp::Node
{
public:
    GlobalLocalizerNode()
        : Node("global_localizer"), map_received_(false), first_odom_received_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Global localizer (MCL) node initialized (C++)");

        // 1. Parameters
        this->declare_parameter("initial_pose_x", 0.0);
        this->declare_parameter("initial_pose_y", 1.0);
        this->declare_parameter("initial_pose_z", 0.5);
        this->declare_parameter("initial_pose_yaw", 0.0);
        this->declare_parameter("min_particles", 500);
        this->declare_parameter("max_particles", 2000);

        init_x_ = this->get_parameter("initial_pose_x").as_double();
        init_y_ = this->get_parameter("initial_pose_y").as_double();
        init_z_ = this->get_parameter("initial_pose_z").as_double();
        double init_yaw = this->get_parameter("initial_pose_yaw").as_double();
        int min_particles = this->get_parameter("min_particles").as_int();
        int max_particles = this->get_parameter("max_particles").as_int();

        // 2. Initialize Particle Filter
        // ParticleFilter 생성자의 파라미터에 맞춰 초기화
        pf_ = std::make_unique<ParticleFilter>(
            min_particles, max_particles, 
            0.2f, 0.2f, 0.2f); // init noise x, y, yaw

        pf_->initialize(static_cast<float>(init_x_), static_cast<float>(init_y_), static_cast<float>(init_yaw));

        // 3. TF Setup
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 4. QoS for Map (Reliable, Transient Local)
        rclcpp::QoS map_qos(rclcpp::KeepLast(1));
        map_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        map_qos.durability(rclcpp::DurabilityPolicy::TransientLocal);

        // 5. Subscribers & Publishers
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", map_qos,
            std::bind(&GlobalLocalizerNode::map_callback, this, std::placeholders::_1));

        initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            std::bind(&GlobalLocalizerNode::initial_pose_callback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&GlobalLocalizerNode::scan_callback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/go1_pose", 10);
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/particle_cloud", 10);
    }

private:
    // --- Member Variables ---
    std::unique_ptr<ParticleFilter> pf_;
    
    // State
    bool map_received_;
    bool first_odom_received_;
    
    double init_x_, init_y_, init_z_;
    
    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Odom caching for motion model
    tf2::Transform last_odom_tf_;

    // ROS handles
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;

    // --- Helper Functions ---

    // 2D 평면(x, y, yaw)만 남기고 나머지(z, roll, pitch)를 제거한 Transform 반환
    tf2::Transform get_2d_transform(const tf2::Transform& src)
    {
        double x = src.getOrigin().x();
        double y = src.getOrigin().y();
        
        double roll, pitch, yaw;
        src.getBasis().getRPY(roll, pitch, yaw);

        tf2::Transform dest;
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        dest.setRotation(q);
        dest.setOrigin(tf2::Vector3(x, y, 0.0));
        return dest;
    }

    // --- Callbacks ---

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received map: %dx%d, res: %.3f", 
                    msg->info.width, msg->info.height, msg->info.resolution);
        
        // nav_msgs::OccupancyGrid의 data는 int8_t 배열이므로 바로 전달 가능
        // ParticleFilter::set_map 시그니처: 
        // void set_map(const std::vector<int8_t>& map_data, int width, int height, float resolution, float origin_x, float origin_y);
        
        pf_->set_map(msg->data, msg->info.width, msg->info.height, 
                     msg->info.resolution, 
                     msg->info.origin.position.x, 
                     msg->info.origin.position.y);
        
        map_received_ = true;
    }

    void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        RCLCPP_INFO(this->get_logger(), "Relocalizing to x:%.2f, y:%.2f, yaw:%.2f", x, y, yaw);
        
        pf_->initialize(static_cast<float>(x), static_cast<float>(y), static_cast<float>(yaw));
        
        // 오도메트리 연속성이 끊기므로 플래그 리셋
        first_odom_received_ = false;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        if (!map_received_) return;

        rclcpp::Time current_time = scan_msg->header.stamp;

        // 1. Get Current Odom Pose (odom -> base)
        geometry_msgs::msg::TransformStamped odom_tf_msg;
        try {
            odom_tf_msg = tf_buffer_->lookupTransform(
                "odom", "base", 
                current_time, 
                rclcpp::Duration::from_seconds(0.1));
        } catch (tf2::TransformException &ex) {
            // 실패 시 최신 데이터 조회 시도
            try {
                odom_tf_msg = tf_buffer_->lookupTransform("odom", "base", rclcpp::Time(0));
            } catch (tf2::TransformException &ex2) {
                RCLCPP_DEBUG(this->get_logger(), "TF lookup failed: %s", ex2.what());
                return;
            }
        }

        tf2::Transform curr_odom_tf_3d;
        tf2::fromMsg(odom_tf_msg.transform, curr_odom_tf_3d);

        // 네비게이션용 2D 오도메트리 생성 (Z, Roll, Pitch 제거)
        tf2::Transform curr_odom_tf_2d = get_2d_transform(curr_odom_tf_3d);

        bool do_update = false;

        // 2. Prediction Step (Motion Model)
        if (first_odom_received_) {
            // T_delta = T_prev_inv * T_curr (Local movement in robot frame)
            tf2::Transform tf_delta = last_odom_tf_.inverse() * curr_odom_tf_2d;

            double dx = tf_delta.getOrigin().x();
            double dy = tf_delta.getOrigin().y();
            
            double droll, dpitch, dyaw;
            tf_delta.getBasis().getRPY(droll, dpitch, dyaw);

            // 이동량이 일정 이상일 때만 업데이트 수행
            if (std::abs(dx) > 0.0001 || std::abs(dy) > 0.0001 || std::abs(dyaw) > 0.0001) {
                pf_->predict(static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dyaw));
                do_update = true;
                last_odom_tf_ = curr_odom_tf_2d;
            }
        } else {
            // 첫 실행 시 초기화
            last_odom_tf_ = curr_odom_tf_2d;
            first_odom_received_ = true;
        }

        if (do_update) {
            // 3. Update Step (Sensor Model)
            // sensor_offset은 base와 laser 사이의 거리. 여기선 0으로 가정 (Python 코드와 동일)
            float sensor_offset[2] = {0.0f, 0.0f};
            
            pf_->update(scan_msg->ranges, 
                        scan_msg->angle_min, 
                        scan_msg->angle_increment, 
                        sensor_offset);
            
            pf_->resample();
        }

        // 4. Get Estimated Pose (Map -> Base)
        std::vector<float> est_pose = pf_->get_estimated_pose(); // [x, y, yaw]

        // 5. Calculate Map -> Odom TF
        // T_map_to_odom = T_map_to_base * (T_odom_to_base)^-1
        
        // 5-1. T_map_to_base 생성
        tf2::Transform tf_map_to_base;
        tf2::Quaternion q_est;
        q_est.setRPY(0, 0, est_pose[2]);
        tf_map_to_base.setRotation(q_est);
        tf_map_to_base.setOrigin(tf2::Vector3(est_pose[0], est_pose[1], 0.0));

        // 5-2. T_odom_to_base의 역행렬 = T_base_to_odom
        // 주의: 반드시 2D화 된 오도메트리를 사용해야 맵이 기울어지지 않음
        tf2::Transform tf_base_to_odom = curr_odom_tf_2d.inverse();

        // 5-3. 최종 T_map_to_odom
        tf2::Transform tf_map_to_odom = tf_map_to_base * tf_base_to_odom;

        // Publish TF
        publish_tf(tf_map_to_odom, scan_msg->header.stamp);

        // 6. Publish Pose & Particles
        publish_mcl_pose(est_pose, scan_msg->header.stamp);

        if (do_update) {
            publish_particles(scan_msg->header.stamp);
        }
    }

    void publish_tf(const tf2::Transform& transform, const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = "map";
        t.child_frame_id = "odom";

        t.transform = tf2::toMsg(transform);
        
        // Map->Odom은 2D 평면상 변환이어야 하므로 Z는 강제로 0 (이미 로직상 0이지만 확실히)
        t.transform.translation.z = 0.0;

        tf_broadcaster_->sendTransform(t);
    }

    void publish_mcl_pose(const std::vector<float>& pose_2d, const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = "map";
        
        msg.pose.position.x = pose_2d[0];
        msg.pose.position.y = pose_2d[1];
        msg.pose.position.z = init_z_; // 초기 설정된 Z값 유지

        tf2::Quaternion q;
        q.setRPY(0, 0, pose_2d[2]);
        msg.pose.orientation = tf2::toMsg(q);

        pose_pub_->publish(msg);
    }

    void publish_particles(const rclcpp::Time& stamp)
    {
        // 구독자가 없으면 연산조차 하지 않음
        if (particle_pub_->get_subscription_count() == 0) return;

        // 성능을 위해 파티클 다운샘플링 (Python 코드와 유사하게 1/10 정도)
        const auto& particles = pf_->get_particles();
        size_t step = std::max(size_t(1), particles.size() / 10);

        geometry_msgs::msg::PoseArray msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = "map";
        
        msg.poses.reserve(particles.size() / step + 1);

        for (size_t i = 0; i < particles.size(); i += step) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = particles[i].x;
            pose.position.y = particles[i].y;
            pose.position.z = 0.0; // 파티클은 2D

            tf2::Quaternion q;
            q.setRPY(0, 0, particles[i].yaw);
            pose.orientation = tf2::toMsg(q);

            msg.poses.push_back(pose);
        }

        particle_pub_->publish(msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GlobalLocalizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
