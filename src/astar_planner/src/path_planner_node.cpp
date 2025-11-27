// astar_planner/src/path_planner_node.cpp

#include <memory>
#include <vector>
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>  // std::pair

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "astar_planner/astar.hpp"

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
  : Node("path_planner_node")
  {
    // Declare parameters
    this->declare_parameter<double>("resolution", 1.0);
    // 장애물 주변 margin [m] (soft cost 영역)
    this->declare_parameter<double>("obstacle_margin", 0.3);

    resolution_ = this->get_parameter("resolution").as_double();
    obstacle_margin_m_ = this->get_parameter("obstacle_margin").as_double();

    // Dynamic parameter change callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&PathPlannerNode::onParameterChange, this, std::placeholders::_1));

    // Initialize flags
    has_map_ = false;
    has_goal_ = false;
    has_current_pose_ = false;
    goal_reached_ = false;

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

    RCLCPP_INFO(this->get_logger(), "Path Planner Node initialized");
    RCLCPP_INFO(this->get_logger(), "Use RViz2 '2D Goal Pose' tool to set a goal");
  }

private:
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

    RCLCPP_INFO(this->get_logger(), "Map received: %dx%d", width, height);
  }

  // ====== Current pose callback ======

  void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!has_current_pose_) {
      has_current_pose_ = true;
      current_pose_ = *msg;
      previous_pose_ = *msg;
      RCLCPP_INFO(this->get_logger(), "Robot position initialized at (%.2f, %.2f)",
        current_pose_.pose.position.x, current_pose_.pose.position.y);
      return;
    }

    // Check if robot position actually changed
    double dx = msg->pose.position.x - previous_pose_.pose.position.x;
    double dy = msg->pose.position.y - previous_pose_.pose.position.y;
    double distance = std::sqrt(dx * dx + dy * dy);

    // Only replan if position changed significantly (moved to new grid cell)
    if (distance < 0.01) {
      return;
    }

    current_pose_ = *msg;

    // Check if goal is reached
    if (has_goal_) {
      double goal_dx = current_pose_.pose.position.x - goal_pose_.pose.position.x;
      double goal_dy = current_pose_.pose.position.y - goal_pose_.pose.position.y;
      double goal_distance = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy);

      if (goal_distance < 0.5) {  // Goal reached threshold
        if (!goal_reached_) {
          RCLCPP_INFO(this->get_logger(), "✓ Goal reached!");
          goal_reached_ = true;
        }
        return;  // Don't replan if goal is reached
      }
    }

    RCLCPP_INFO(this->get_logger(), "Robot moved to (%.2f, %.2f)",
      current_pose_.pose.position.x, current_pose_.pose.position.y);

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
    goal_pose_ = *msg;
    has_goal_ = true;
    goal_reached_ = false;  // Reset goal reached flag for new goal

    RCLCPP_INFO(this->get_logger(),
      "New goal received: (%.2f, %.2f)",
      goal_pose_.pose.position.x,
      goal_pose_.pose.position.y);

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

    // Convert world coordinates to grid coordinates
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

    // Convert to ROS Path message
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "map";

    // First waypoint: 현재 로봇 pose
    geometry_msgs::msg::PoseStamped first_pose;
    first_pose.header = path_msg.header;
    first_pose.pose = current_pose_.pose;
    path_msg.poses.push_back(first_pose);

    // 나머지 경로 점들을 Pose로 추가
    geometry_msgs::msg::PoseStamped pose;
    for (size_t i = 0; i < smooth_world_path.size(); ++i) {
      double wx = smooth_world_path[i].first;
      double wy = smooth_world_path[i].second;

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
      pose.pose.orientation.w = 1.0;

      path_msg.poses.push_back(pose);
    }

    path_pub_->publish(path_msg);

    // Publish visualization markers (smooth path 기준)
    publishPathMarkers(smooth_world_path);

    // Only log if path length changed significantly or first time
    static size_t last_path_size = 0;
    if (last_path_size == 0 ||
        std::abs((int)smooth_world_path.size() - (int)last_path_size) > 3) {
      RCLCPP_INFO(this->get_logger(), "Path updated: %zu waypoints (smoothed)",
        smooth_world_path.size());
      last_path_size = smooth_world_path.size();
    }
  }

  // ====== Coordinate transforms ======

  astar_planner::GridCell worldToGrid(double x, double y)
  {
    astar_planner::GridCell cell;

    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = map_msg_->info.resolution;

    cell.x = static_cast<int>((x - origin_x) / resolution);
    cell.y = static_cast<int>((y - origin_y) / resolution);

    return cell;
  }

  std::pair<double, double> gridToWorld(int x, int y)
  {
    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = map_msg_->info.resolution;

    double world_x = origin_x + (x + 0.5) * resolution;
    double world_y = origin_y + (y + 0.5) * resolution;

    return {world_x, world_y};
  }

  // ====== Visualization ======

  void publishPathMarkers(const std::vector<std::pair<double,double>> & path_world)
  {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
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
    marker.header.frame_id = "map";
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
      }
    }

    return result;
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

    // margin [m] → grid 셀 수로 변환
    int inflation_cells = static_cast<int>(
      std::round(obstacle_margin_m_ / map_resolution));

    // planning map: 0 = free, 1 = margin zone, 2 = real obstacle
    map_grid_.assign(height, std::vector<int>(width, 0));

    // 1단계: real obstacle → 2로 설정
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (raw_map_grid_[y][x] == 1) {
          map_grid_[y][x] = 2;  // real obstacle
        }
      }
    }

    // 2단계: margin zone 설정 (real obstacle 주변을 1로)
    if (inflation_cells > 0) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (raw_map_grid_[y][x] == 1) {
            for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
              for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
                int ny = y + dy;
                int nx = x + dx;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                  continue;
                }
                // 아직 real obstacle(2)이 아닌 free(0)만 margin(1)으로 올리기
                if (map_grid_[ny][nx] == 0) {
                  map_grid_[ny][nx] = 1;
                }
              }
            }
          }
        }
      }
    }

    astar_.setMap(map_grid_);

    RCLCPP_INFO(this->get_logger(),
      "Updated planning map with obstacle_margin = %.3f m (~%d cells)",
      obstacle_margin_m_, inflation_cells);
  }

  // ====== ROS objects ======

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;

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
  double resolution_;        // kept for compatibility
  double obstacle_margin_m_; // safety margin [m] around obstacles (soft cost)

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
