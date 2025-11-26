#include <memory>
#include <vector>
#include <chrono>
#include <cmath>
#include <functional>

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

class PathPlannerNode : public rclcpp::Node
{
public:
  PathPlannerNode()
  : Node("path_planner_node")
  {
    // Declare parameters
    this->declare_parameter<double>("resolution", 1.0);
    // 장애물 주변 margin [m] (soft cost 영역)
    this->declare_parameter<double>("obstacle_margin", 1.5);

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
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", 10,
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

    // Convert grid path to ROS Path message
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "map";

    // First waypoint is always current pose
    geometry_msgs::msg::PoseStamped first_pose;
    first_pose.header.stamp = this->now();
    first_pose.header.frame_id = "map";
    first_pose.pose = current_pose_.pose;
    path_msg.poses.push_back(first_pose);

    // Add rest of the path (skip first cell if it's same as current position)
    for (size_t i = 0; i < path_cells.size(); ++i) {
      const auto & cell = path_cells[i];

      auto world_pos = gridToWorld(cell.x, cell.y);

      // Skip if this waypoint is too close to current position
      double dx = world_pos.first - current_pose_.pose.position.x;
      double dy = world_pos.second - current_pose_.pose.position.y;
      double dist = std::sqrt(dx * dx + dy * dy);

      if (i == 0 && dist < 0.3) {
        continue;  // Skip first cell if robot is already there
      }

      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp = this->now();
      pose.header.frame_id = "map";
      pose.pose.position.x = world_pos.first;
      pose.pose.position.y = world_pos.second;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;

      path_msg.poses.push_back(pose);
    }

    path_pub_->publish(path_msg);

    // Publish visualization markers
    publishPathMarkers(path_cells);

    // Only log if path length changed significantly or first time
    static size_t last_path_size = 0;
    if (last_path_size == 0 || std::abs((int)path_cells.size() - (int)last_path_size) > 3) {
      RCLCPP_INFO(this->get_logger(), "Path updated: %zu waypoints", path_cells.size());
      last_path_size = path_cells.size();
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

  void publishPathMarkers(const std::vector<astar_planner::GridCell> & path)
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

    for (const auto & cell : path) {
      geometry_msgs::msg::Point p;
      auto world_pos = gridToWorld(cell.x, cell.y);
      p.x = world_pos.first;
      p.y = world_pos.second;
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
      // fallback: 그냥 raw 맵을 0/2로만 써도 되지만, 여기선 안전하게 종료
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
