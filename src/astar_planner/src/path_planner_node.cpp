// astar_planner/src/path_planner_node.cpp

#include <memory>
#include <vector>
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>  // std::pair
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "astar_planner/astar.hpp"

// TF2
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;

// ======================
// Catmull–Rom Spline helper (전역 helper 함수)
// ======================
namespace
{

std::pair<double, double> catmullRomPoint(
    const std::pair<double,double>& p0,
    const std::pair<double,double>& p1,
    const std::pair<double,double>& p2,
    const std::pair<double,double>& p3,
    double t)
{
  double t2 = t * t;
  double t3 = t2 * t;

  double x = 0.5 * (2 * p1.first +
      (-p0.first + p2.first) * t +
      (2 * p0.first - 5 * p1.first + 4 * p2.first - p3.first) * t2 +
      (-p0.first + 3 * p1.first - 3 * p2.first + p3.first) * t3);

  double y = 0.5 * (2 * p1.second +
      (-p0.second + p2.second) * t +
      (2 * p0.second - 5 * p1.second + 4 * p2.second - p3.second) * t2 +
      (-p0.second + 3 * p1.second - 3 * p2.second + p3.second) * t3);

  return {x, y};
}

// pts: world 좌표 (x,y) 리스트
// 반환: Catmull–Rom으로 보간한 더 촘촘한 경로
std::vector<std::pair<double,double>>
smoothPathCatmullRom(const std::vector<std::pair<double,double>>& pts)
{
  std::vector<std::pair<double,double>> smooth;

  if (pts.size() < 4) {
    return pts;  // 점이 4개 미만이면 그대로 반환
  }

  // 구간마다 보간
  for (size_t i = 0; i + 3 < pts.size(); ++i) {
    const auto& p0 = pts[i];
    const auto& p1 = pts[i+1];
    const auto& p2 = pts[i+2];
    const auto& p3 = pts[i+3];

    // t step: 곡선 해상도 (0.05면 구간당 20개 샘플)
    for (double t = 0.0; t <= 1.0; t += 0.05) {
      smooth.push_back(catmullRomPoint(p0, p1, p2, p3, t));
    }
  }

  // 마지막 점을 확실히 포함
  smooth.push_back(pts.back());

  return smooth;
}

} // namespace

// ======================
// PathPlannerNode 정의
// ======================

class PathPlannerNode : public rclcpp::Node
{
public:
  PathPlannerNode()
  : Node("path_planner_node"),
    planning_resolution_m_(0.25),  // default coarse resolution for faster planning
    goal_yaw_(0.0),
    downsample_factor_(1),
    global_frame_("map"),          
    effective_resolution_m_(1.0)
  {
    // Declare parameters
    this->declare_parameter<double>("resolution", 1.0);
    this->declare_parameter<double>("obstacle_margin", 0.0);
    this->declare_parameter<double>("planning_resolution", planning_resolution_m_);

    resolution_ = this->get_parameter("resolution").as_double();
    obstacle_margin_m_ = this->get_parameter("obstacle_margin").as_double();
    planning_resolution_m_ = this->get_parameter("planning_resolution").as_double();

    // Dynamic parameter change callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&PathPlannerNode::onParameterChange, this, std::placeholders::_1));

    // Initialize flags
    has_map_ = false;
    has_goal_ = false;
    has_current_pose_ = false;
    goal_reached_ = false;

    // TF buffer & listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscribers
    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", map_qos,
      std::bind(&PathPlannerNode::mapCallback, this, std::placeholders::_1));

    current_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/go1_pose", 10,
      std::bind(&PathPlannerNode::currentPoseCallback, this, std::placeholders::_1));

    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      std::bind(&PathPlannerNode::goalCallback, this, std::placeholders::_1));

    // Publishers
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/local_path", 10);
    viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/path_markers", 10);
    goal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);
    goal_reached_pub_ = this->create_publisher<std_msgs::msg::Bool>("/goal_reached", 10);
    goal_state_timer_ = this->create_wall_timer(
      1s, std::bind(&PathPlannerNode::publishGoalReachedState, this));

    RCLCPP_INFO(this->get_logger(), "Path Planner Node initialized");
    RCLCPP_INFO(this->get_logger(), "Use RViz2 '2D Goal Pose' tool to set a goal");
  }

private:
  // ======================
  // Quaternion ↔ yaw 변환 helper
  // ======================
  double quaternionToYaw(const geometry_msgs::msg::Quaternion & q)
  {
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
  {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(yaw / 2.0);
    q.w = std::cos(yaw / 2.0);
    return q;
  }

  double normalizeAngle(double a)
  {
    const double PI = 3.14159265358979323846;
    const double TWO_PI = 2.0 * PI;
    while (a > PI)  { a -= TWO_PI; }
    while (a < -PI) { a += TWO_PI; }
    return a;
  }

  double shortestAngularDistance(double from, double to)
  {
    return normalizeAngle(to - from);
  }

  // ======================
  // TF helper: 임의 frame → global_frame_ (기본: "map")
  // ======================
  bool transformToMap(
      const geometry_msgs::msg::PoseStamped & in,
      geometry_msgs::msg::PoseStamped & out)
  {
    // frame_id가 비어있거나 이미 global_frame_이면 그대로 사용
    if (in.header.frame_id.empty() || in.header.frame_id == global_frame_) {
      out = in;
      out.header.frame_id = global_frame_;
      return true;
    }

    try {
      out = tf_buffer_->transform(in, global_frame_);
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(),
        "Failed to transform from '%s' to '%s': %s",
        in.header.frame_id.c_str(), global_frame_.c_str(), ex.what());
      return false;
    }
  }

  // ====== Map callback ======

  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    map_msg_ = msg;

    int width  = static_cast<int>(msg->info.width);
    int height = static_cast<int>(msg->info.height);

    raw_map_grid_.clear();
    raw_map_grid_.resize(height, std::vector<int>(width, 0));

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int index = y * width + x;
        // OccupancyGrid: -1 (unknown), 0 (free), 100 (occupied)
        // raw_map_grid_: 0 = free, 1 = real obstacle
        if (msg->data[index] > 50 || msg->data[index] < 0) {
          raw_map_grid_[y][x] = 1;  // real obstacle
        } else {
          raw_map_grid_[y][x] = 0;  // free
        }
      }
    }

    has_map_ = true;

    // 현재 obstacle_margin 설정값을 이용해 planning용 맵 생성
    updateInflatedMap();

    RCLCPP_INFO(this->get_logger(), "Map received: %dx%d (frame_id=%s)",
      width, height, msg->header.frame_id.c_str());
  }

  // ====== Current pose callback ======

  void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // 어떤 frame에서 오든 global_frame_ ("map") 기준으로 변환
    geometry_msgs::msg::PoseStamped pose_map;
    if (!transformToMap(*msg, pose_map)) {
      return;  // 변환 실패 시 이번 콜백은 패스
    }

    if (!has_current_pose_) {
      has_current_pose_ = true;
      current_pose_ = pose_map;
      previous_pose_ = pose_map;
      RCLCPP_INFO(this->get_logger(), "Robot position initialized at (%.2f, %.2f) in %s",
        current_pose_.pose.position.x, current_pose_.pose.position.y,
        global_frame_.c_str());
      return;
    }

    // Check if robot position actually changed
    double dx = pose_map.pose.position.x - previous_pose_.pose.position.x;
    double dy = pose_map.pose.position.y - previous_pose_.pose.position.y;
    double distance = std::sqrt(dx * dx + dy * dy);

    // Only replan if position changed significantly (moved to new grid cell)
    if (distance < 0.01) {
      return;
    }

    current_pose_ = pose_map;

    // Check if goal is reached (위치는 여기서만 체크)
    if (has_goal_) {
      double goal_dx = current_pose_.pose.position.x - goal_pose_.pose.position.x;
      double goal_dy = current_pose_.pose.position.y - goal_pose_.pose.position.y;
      double goal_distance = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy);

      if (goal_distance < 0.1) {  // Goal reached threshold (위치 기준)
        if (!goal_reached_) {
          RCLCPP_INFO(this->get_logger(), "✓ Goal position reached (within 0.5 m)");
          goal_reached_ = true;
          publishGoalReached(true);
          publishEmptyPath();  // clear remaining path so tracker does not keep moving
        }
        // 위치 도착 후에는 새로 replan 하지 않음
        return;
      }
    }

    RCLCPP_INFO(this->get_logger(), "Robot moved to (%.2f, %.2f) in %s",
      current_pose_.pose.position.x, current_pose_.pose.position.y,
      global_frame_.c_str());

    // Store current position as previous for next comparison
    previous_pose_ = current_pose_;

    // Replan path whenever robot position changes (and we have a goal)
    if (has_map_ && has_goal_ && !goal_reached_) {
      replanPath();
    }
  }

  // ====== Goal callback ======

  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // goal도 어떤 frame에서 오든 global_frame_으로 변환
    geometry_msgs::msg::PoseStamped goal_map;
    if (!transformToMap(*msg, goal_map)) {
      return;
    }

    goal_pose_ = goal_map;
    has_goal_ = true;
    goal_reached_ = false;  // Reset goal reached flag for new goal
    publishGoalReached(false);

    // goal yaw 업데이트
    goal_yaw_ = quaternionToYaw(goal_pose_.pose.orientation);
    const double rad2deg = 180.0 / 3.14159265358979323846;

    RCLCPP_INFO(this->get_logger(),
      "New goal received: (%.2f, %.2f) in %s, yaw=%.2f deg",
      goal_pose_.pose.position.x,
      goal_pose_.pose.position.y,
      global_frame_.c_str(),
      goal_yaw_ * rad2deg);

    // Publish goal marker for visualization
    publishGoalMarker();

    // Plan path immediately when goal is set
    if (has_map_ && has_current_pose_) {
      replanPath();
    }
  }

  // ====== Path planning ======

  void replanPath()
  {
    if (!has_map_ || !has_current_pose_ || !has_goal_) {
      return;
    }

    // Convert world coordinates (global_frame_) to grid coordinates
    astar_planner::GridCell start = worldToGrid(
      current_pose_.pose.position.x,
      current_pose_.pose.position.y);

    astar_planner::GridCell goal = worldToGrid(
      goal_pose_.pose.position.x,
      goal_pose_.pose.position.y);

    // Find path using A*
    auto path_cells = astar_.findPath(start, goal);

    if (path_cells.empty()) {
      RCLCPP_WARN(this->get_logger(), "No path found!");
      return;
    }

    // Grid path → world 좌표 리스트로 변환
    std::vector<std::pair<double,double>> world_path;
    world_path.reserve(path_cells.size());

    for (const auto & cell : path_cells) {
      auto wp = gridToWorld(cell.x, cell.y);
      world_path.push_back(wp);
    }

    // Catmull–Rom spline smoothing 적용
    std::vector<std::pair<double,double>> smooth_world_path;
    if (world_path.size() >= 4) {
      smooth_world_path = smoothPathCatmullRom(world_path);
    } else {
      smooth_world_path = world_path;
    }

    // =====================================================
    //  1) smooth_world_path로 목표 지점까지 이동
    //  2) 목표 지점 (gx,gy)에서 제자리 회전 경로를 추가 (x,y 동일, yaw만 변화)
    // =====================================================

    std::vector<std::pair<double,double>> final_path;  // 위치
    std::vector<double> yaw_list;                      // 각 위치에서의 yaw

    if (!smooth_world_path.empty()) {
      const std::size_t N_move = smooth_world_path.size();
      final_path.reserve(N_move + 20);
      yaw_list.reserve(N_move + 20);

      // 이동 파트 마지막 구간의 방향 (goal에 들어갈 때의 heading)
      double yaw_move_end = 0.0;
      if (N_move >= 2) {
        const auto &p_prev = smooth_world_path[N_move - 2];
        const auto &p_last = smooth_world_path[N_move - 1];
        yaw_move_end = std::atan2(
          p_last.second - p_prev.second,
          p_last.first  - p_prev.first);
      } else {
        // 경로가 한 점 뿐이면 현재 로봇 yaw 사용
        yaw_move_end = quaternionToYaw(current_pose_.pose.orientation);
      }

      // 1) 이동 파트: smooth_world_path 그대로 사용, yaw는 접선 방향
      for (std::size_t i = 0; i < N_move; ++i) {
        final_path.push_back(smooth_world_path[i]);

        double yaw_i = yaw_move_end;
        if (i + 1 < N_move) {
          const auto &p     = smooth_world_path[i];
          const auto &p_next= smooth_world_path[i+1];
          yaw_i = std::atan2(
            p_next.second - p.second,
            p_next.first  - p.first);
        }
        yaw_list.push_back(yaw_i);
      }

      // 2) 제자리 회전 파트
      const int N_rot = 20;  // 회전을 몇 단계로 나눌지 (원하면 튜닝 가능)
      double gx = goal_pose_.pose.position.x;
      double gy = goal_pose_.pose.position.y;

      // yaw_move_end → goal_yaw_ 로 가는 최단 회전 방향
      double diff = shortestAngularDistance(yaw_move_end, goal_yaw_);

      for (int k = 1; k <= N_rot; ++k) {
        double alpha = static_cast<double>(k) / static_cast<double>(N_rot);
        double yaw_k = normalizeAngle(yaw_move_end + alpha * diff);

        // 위치는 모두 goal 지점 (제자리 회전)
        final_path.emplace_back(gx, gy);
        yaw_list.push_back(yaw_k);
      }
    }

    if (final_path.empty()) {
      RCLCPP_WARN(this->get_logger(), "Final path is empty after processing.");
      return;
    }

    // Convert to ROS Path message
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = global_frame_;  // 일관된 global frame

    // First waypoint: 현재 로봇 pose (현재 orientation 그대로)
    geometry_msgs::msg::PoseStamped first_pose;
    first_pose.header = path_msg.header;
    first_pose.pose = current_pose_.pose;
    path_msg.poses.push_back(first_pose);

    // 나머지 경로 점들을 Pose로 추가 (position + orientation)
    geometry_msgs::msg::PoseStamped pose;
    for (std::size_t i = 0; i < final_path.size(); ++i) {
      double wx = final_path[i].first;
      double wy = final_path[i].second;

      // 첫 점이 현재 위치와 너무 가까우면 생략
      double dx = wx - current_pose_.pose.position.x;
      double dy = wy - current_pose_.pose.position.y;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (i == 0 && dist < 0.1) {
        continue;
      }

      pose.header = path_msg.header;
      pose.pose.position.x = wx;
      pose.pose.position.y = wy;
      pose.pose.position.z = 0.0;

      double yaw = (i < yaw_list.size()) ? yaw_list[i] : goal_yaw_;
      pose.pose.orientation = yawToQuaternion(yaw);

      path_msg.poses.push_back(pose);
    }

    path_pub_->publish(path_msg);

    // 시각화는 최종 경로(final_path) 기준으로
    publishPathMarkers(final_path);

    // Only log if path length changed significantly or first time
    static std::size_t last_path_size = 0;
    if (last_path_size == 0 ||
        std::abs(static_cast<int>(final_path.size()) -
                 static_cast<int>(last_path_size)) > 3) {
      RCLCPP_INFO(this->get_logger(), "Path updated: %zu waypoints (including in-place rotation)",
        final_path.size());
      last_path_size = final_path.size();
    }
  }

  // ====== Coordinate transforms (map grid ↔ world) ======

  astar_planner::GridCell worldToGrid(double x, double y)
  {
    astar_planner::GridCell cell;

    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = effective_resolution_m_;

    cell.x = static_cast<int>((x - origin_x) / resolution);
    cell.y = static_cast<int>((y - origin_y) / resolution);

    return cell;
  }

  std::pair<double, double> gridToWorld(int x, int y)
  {
    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = effective_resolution_m_;

    double world_x = origin_x + (x + 0.5) * resolution;
    double world_y = origin_y + (y + 0.5) * resolution;

    return {world_x, world_y};
  }

  // ====== Visualization ======

  void publishPathMarkers(const std::vector<std::pair<double,double>> & path_world)
  {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = global_frame_;
    line_marker.header.stamp = this->now();
    line_marker.ns = "path";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.1;  // Line width
    line_marker.color.r = 0.0;
    line_marker.color.g = 1.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 1.0;

    for (const auto & wp : path_world) {
      geometry_msgs::msg::Point p;
      p.x = wp.first;
      p.y = wp.second;
      p.z = 0.1;
      line_marker.points.push_back(p);
    }

    marker_array.markers.push_back(line_marker);
    viz_pub_->publish(marker_array);
  }

  void publishGoalMarker()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = global_frame_;
    marker.header.stamp = this->now();
    marker.ns = "goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = goal_pose_.pose.position.x;
    marker.pose.position.y = goal_pose_.pose.position.y;
    marker.pose.position.z = 0.5;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.8;
    marker.scale.y = 0.8;
    marker.scale.z = 0.8;

    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 0.8;

    goal_marker_pub_->publish(marker);
  }

  // ====== Dynamic parameters & map inflation ======

  rcl_interfaces::msg::SetParametersResult
  onParameterChange(const std::vector<rclcpp::Parameter> & params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    for (const auto & param : params) {
      if (param.get_name() == "obstacle_margin") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "obstacle_margin must be non-negative";
          return result;
        }

        obstacle_margin_m_ = value;
        RCLCPP_INFO(this->get_logger(),
          "Updated obstacle_margin to %.3f m", obstacle_margin_m_);

        if (has_map_) {
          updateInflatedMap();
        }
      } else if (param.get_name() == "planning_resolution") {
        double value = param.as_double();
        if (value <= 0.0) {
          result.successful = false;
          result.reason = "planning_resolution must be positive";
          return result;
        }
        planning_resolution_m_ = value;
        RCLCPP_INFO(this->get_logger(),
          "Updated planning_resolution to %.3f m", planning_resolution_m_);
        if (has_map_) {
          updateInflatedMap();
        }
      }
    }

    return result;
  }

  void publishGoalReached(bool reached)
  {
    std_msgs::msg::Bool msg;
    msg.data = reached;
    goal_reached_pub_->publish(msg);
  }

  void publishGoalReachedState()
  {
    publishGoalReached(goal_reached_);
  }

  void publishEmptyPath()
  {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = global_frame_;
    path_pub_->publish(path_msg);
  }

  void updateInflatedMap()
  {
    if (!has_map_ || !map_msg_ || raw_map_grid_.empty()) {
      return;
    }

    int width  = static_cast<int>(map_msg_->info.width);
    int height = static_cast<int>(map_msg_->info.height);
    double map_resolution = map_msg_->info.resolution;

    if (width <= 0 || height <= 0 || map_resolution <= 0.0) {
      // fallback: 안전하게 종료
      return;
    }

    // Downsample factor to speed up planning (coarser grid)
    downsample_factor_ = std::max(
      1, static_cast<int>(std::round(planning_resolution_m_ / map_resolution)));
    effective_resolution_m_ = map_resolution * static_cast<double>(downsample_factor_);

    int coarse_width = static_cast<int>(std::ceil(
      static_cast<double>(width) / static_cast<double>(downsample_factor_)));
    int coarse_height = static_cast<int>(std::ceil(
      static_cast<double>(height) / static_cast<double>(downsample_factor_)));

    // planning map: 0 = free, 2 = real obstacle (after downsampling)
    map_grid_.assign(coarse_height, std::vector<int>(coarse_width, 0));

    // Step 1: block-wise OR to project fine grid onto coarse grid
    for (int cy = 0; cy < coarse_height; ++cy) {
      int y_start = cy * downsample_factor_;
      int y_end = std::min(height, y_start + downsample_factor_);
      for (int cx = 0; cx < coarse_width; ++cx) {
        int x_start = cx * downsample_factor_;
        int x_end = std::min(width, x_start + downsample_factor_);

        bool has_obstacle = false;
        for (int y = y_start; y < y_end && !has_obstacle; ++y) {
          for (int x = x_start; x < x_end; ++x) {
            if (raw_map_grid_[y][x] == 1) {
              has_obstacle = true;
              break;
            }
          }
        }

        if (has_obstacle) {
          map_grid_[cy][cx] = 2;  // coarse obstacle
        }
      }
    }

    // Step 2: inflate obstacles according to obstacle_margin (in coarse grid units)
    int inflation_cells = static_cast<int>(
      std::round(obstacle_margin_m_ / effective_resolution_m_));
    if (inflation_cells > 0) {
      std::vector<std::vector<int>> inflated = map_grid_;
      for (int y = 0; y < coarse_height; ++y) {
        for (int x = 0; x < coarse_width; ++x) {
          if (map_grid_[y][x] != 2) {
            continue;
          }
          for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
            for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
              int nx = x + dx;
              int ny = y + dy;
              if (nx < 0 || nx >= coarse_width || ny < 0 || ny >= coarse_height) {
                continue;
              }
              inflated[ny][nx] = 2;
            }
          }
        }
      }
      map_grid_.swap(inflated);
    }

    // A*에 맵 해상도[m/셀] 전달
    astar_.setResolution(effective_resolution_m_);

    // planning map 전달 (0: free, 1: margin, 2: real obstacle)
    astar_.setMap(map_grid_);

    RCLCPP_INFO(this->get_logger(),
      "Updated planning map. raw_res=%.3f m, planning_res=%.3f m, obstacle_margin=%.3f m (~%d coarse cells)",
      map_resolution, effective_resolution_m_, obstacle_margin_m_, inflation_cells);
  }

  // ====== ROS objects ======

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_reached_pub_;
  rclcpp::TimerBase::SharedPtr goal_state_timer_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  

  // State variables
  bool has_map_;
  bool has_goal_;
  bool has_current_pose_;
  bool goal_reached_;

  nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;
  geometry_msgs::msg::PoseStamped current_pose_;
  geometry_msgs::msg::PoseStamped previous_pose_;
  geometry_msgs::msg::PoseStamped goal_pose_;

  // Map data: raw (0/1) and planning (0/1/2)
  std::vector<std::vector<int>> raw_map_grid_;  // 0: free, 1: real obstacle
  std::vector<std::vector<int>> map_grid_;      // 0: free, 1: margin, 2: real obstacle
  astar_planner::AStar astar_;

  // Parameters
  double resolution_;             // kept for compatibility
  double obstacle_margin_m_;      // safety margin [m] around obstacles (soft cost)
  double planning_resolution_m_;  // desired planning grid resolution [m]

  // Goal yaw (rad)
  double goal_yaw_;

  // Downsample info
  int downsample_factor_;
  double effective_resolution_m_;

  // Global frame (TF)
  std::string global_frame_;  // 보통 "map"

  // Dynamic parameter callback
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
