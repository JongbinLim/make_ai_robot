#include "astar_planner/astar.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

namespace astar_planner
{

AStar::AStar()
: map_width_(0), map_height_(0)
{
}

AStar::~AStar()
{
}

void AStar::setMap(const std::vector<std::vector<int>>& map)
{
  // map 값 의미:
  // 0 = free
  // 1 = margin zone (soft cost 영역, 통과 가능하지만 penalty 부과)
  // 2 = real obstacle (통과 불가)
  map_ = map;
  if (!map_.empty()) {
    map_height_ = map_.size();
    map_width_ = map_[0].size();
  } else {
    map_height_ = 0;
    map_width_ = 0;
  }
}

double AStar::calculateHeuristic(const GridCell& a, const GridCell& b) const
{
  // Euclidean distance
  double dx = static_cast<double>(a.x - b.x);
  double dy = static_cast<double>(a.y - b.y);
  return std::sqrt(dx * dx + dy * dy);
}

bool AStar::isValid(const GridCell& cell) const
{
  if (cell.x < 0 || cell.x >= map_width_ || cell.y < 0 || cell.y >= map_height_) {
    return false;
  }
  int v = map_[cell.y][cell.x];
  // 2 = real obstacle만 통과 불가, 0/1은 통과 가능
  return v != 2;
}

std::vector<GridCell> AStar::getNeighbors(const GridCell& cell) const
{
  std::vector<GridCell> neighbors;

  // 8-connected grid: up, down, left, right, and 4 diagonals
  std::vector<std::pair<int, int>> directions = {
    {0, 1},   // up
    {0, -1},  // down
    {1, 0},   // right
    {-1, 0},  // left
    {1, 1},   // up-right
    {1, -1},  // down-right
    {-1, 1},  // up-left
    {-1, -1}  // down-left
  };

  for (const auto& dir : directions) {
    GridCell neighbor = {cell.x + dir.first, cell.y + dir.second};
    if (isValid(neighbor)) {
      neighbors.push_back(neighbor);
    }
  }

  return neighbors;
}

std::vector<GridCell> AStar::reconstructPath(
  const std::unordered_map<GridCell, GridCell, GridCellHash>& came_from,
  const GridCell& start,
  const GridCell& goal) const
{
  std::vector<GridCell> path;
  GridCell current = goal;

  while (!(current == start)) {
    path.push_back(current);
    auto it = came_from.find(current);
    if (it == came_from.end()) {
      break;
    }
    current = it->second;
  }

  path.push_back(start);
  std::reverse(path.begin(), path.end());

  return path;
}

std::vector<GridCell> AStar::findPath(const GridCell& start, const GridCell& goal)
{
  std::vector<GridCell> empty_path;

  // Check if start and goal are valid (real obstacle 위에 있으면 안 됨)
  if (!isValid(start)) {
    std::cerr << "Start position is invalid or occupied!" << std::endl;
    return empty_path;
  }

  if (!isValid(goal)) {
    std::cerr << "Goal position is invalid or occupied!" << std::endl;
    return empty_path;
  }

  // Priority queue for open set
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;

  // Track visited nodes
  std::unordered_map<GridCell, bool, GridCellHash> closed_set;

  // Track g_cost for each node
  std::unordered_map<GridCell, double, GridCellHash> g_score;

  // Track parent of each node
  std::unordered_map<GridCell, GridCell, GridCellHash> came_from;

  // Initialize start node
  Node start_node;
  start_node.cell = start;
  start_node.g_cost = 0.0;
  start_node.h_cost = calculateHeuristic(start, goal);
  start_node.f_cost = start_node.g_cost + start_node.h_cost;
  start_node.parent = start;

  open_set.push(start_node);
  g_score[start] = 0.0;

  // margin penalty (soft cost): margin 셀을 지날 때 추가로 더해줄 비용
  const double margin_penalty = 2.0;  // 필요하면 나중에 튜닝 가능

  while (!open_set.empty()) {
    // Get node with lowest f_cost
    Node current = open_set.top();
    open_set.pop();

    // Check if we reached the goal
    if (current.cell == goal) {
      return reconstructPath(came_from, start, goal);
    }

    // Skip if already processed
    if (closed_set[current.cell]) {
      continue;
    }

    closed_set[current.cell] = true;

    // Check all neighbors
    std::vector<GridCell> neighbors = getNeighbors(current.cell);

    for (const auto& neighbor : neighbors) {
      // Skip if already processed
      if (closed_set[neighbor]) {
        continue;
      }

      // Calculate movement cost (1 for straight, sqrt(2) for diagonal)
      double dx = static_cast<double>(neighbor.x - current.cell.x);
      double dy = static_cast<double>(neighbor.y - current.cell.y);
      double movement_cost = std::sqrt(dx * dx + dy * dy);

      // margin 구역 penalty 추가
      double penalty = 0.0;
      int cell_value = map_[neighbor.y][neighbor.x];
      if (cell_value == 1) {
        penalty = margin_penalty;
      }

      double tentative_g = current.g_cost + movement_cost + penalty;

      // Check if this path is better
      auto it = g_score.find(neighbor);
      if (it == g_score.end() || tentative_g < it->second) {
        // This path is better, record it
        came_from[neighbor] = current.cell;
        g_score[neighbor] = tentative_g;

        Node neighbor_node;
        neighbor_node.cell = neighbor;
        neighbor_node.g_cost = tentative_g;
        neighbor_node.h_cost = calculateHeuristic(neighbor, goal);
        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost;
        neighbor_node.parent = current.cell;

        open_set.push(neighbor_node);
      }
    }
  }

  std::cerr << "No path found!" << std::endl;
  return empty_path;
}

}  // namespace astar_planner
