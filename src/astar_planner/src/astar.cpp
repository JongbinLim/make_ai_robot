#include "astar_planner/astar.hpp"

#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <queue>

namespace astar_planner
{

AStar::AStar()
: map_width_(0),
  map_height_(0),
  max_penalty_(20.0),              // 장애물 바로 옆 penalty
  max_influence_dist_cells_(8.0)   // 장애물에서 8셀까지 penalty 영향
{
}

AStar::~AStar()
{
}

void AStar::setMap(const std::vector<std::vector<int>> & map)
{
  // map 값 의미:
  // 0 = free
  // 1 = margin zone (soft cost 영역, 통과 가능하지만 penalty 부과)
  // 2 = real obstacle (통과 불가)
  map_ = map;
  if (!map_.empty()) {
    map_height_ = static_cast<int>(map_.size());
    map_width_ = static_cast<int>(map_[0].size());
  } else {
    map_height_ = 0;
    map_width_ = 0;
  }

  // ---------- 장애물까지 거리 맵(distance field) 계산 ----------
  const double INF = std::numeric_limits<double>::infinity();
  distance_map_.assign(
    map_height_, std::vector<double>(map_width_, INF));

  std::queue<GridCell> q;

  // 1) real obstacle(값=2)인 셀들을 seed로 넣고 거리 0으로 초기화
  for (int y = 0; y < map_height_; ++y) {
    for (int x = 0; x < map_width_; ++x) {
      if (map_[y][x] == 2) {  // real obstacle
        distance_map_[y][x] = 0.0;
        q.push(GridCell{x, y});
      }
    }
  }

  // 2) 4-connected BFS 로 각 셀까지의 "장애물까지 최소 거리(셀)" 계산
  const std::vector<std::pair<int, int>> dirs = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}
  };

  while (!q.empty()) {
    GridCell cur = q.front();
    q.pop();
    double cur_dist = distance_map_[cur.y][cur.x];

    for (const auto & d : dirs) {
      int nx = cur.x + d.first;
      int ny = cur.y + d.second;
      if (nx < 0 || nx >= map_width_ || ny < 0 || ny >= map_height_) {
        continue;
      }

      // 장애물 셀은 0으로 유지
      if (map_[ny][nx] == 2) {
        continue;
      }

      double nd = cur_dist + 1.0;  // 한 칸 멀어질 때마다 +1 셀
      if (nd < distance_map_[ny][nx]) {
        distance_map_[ny][nx] = nd;
        q.push(GridCell{nx, ny});
      }
    }
  }

  // 이제 distance_map_[y][x]에는
  // "가장 가까운 real obstacle까지의 거리(셀)" 이 들어 있음
}

double AStar::calculateHeuristic(const GridCell & a, const GridCell & b) const
{
  // Euclidean distance
  double dx = static_cast<double>(a.x - b.x);
  double dy = static_cast<double>(a.y - b.y);
  return std::sqrt(dx * dx + dy * dy);
}

bool AStar::isValid(const GridCell & cell) const
{
  if (cell.x < 0 || cell.x >= map_width_ || cell.y < 0 || cell.y >= map_height_) {
    return false;
  }
  int v = map_[cell.y][cell.x];
  // 2 = real obstacle만 통과 불가, 0/1은 통과 가능
  return v != 2;
}

std::vector<GridCell> AStar::getNeighbors(const GridCell & cell) const
{
  std::vector<GridCell> neighbors;

  // 8-connected grid: 상하좌우 + 대각선
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

  for (const auto & dir : directions) {
    GridCell neighbor{cell.x + dir.first, cell.y + dir.second};
    if (isValid(neighbor)) {
      neighbors.push_back(neighbor);
    }
  }

  return neighbors;
}

std::vector<GridCell> AStar::reconstructPath(
  const std::unordered_map<GridCell, GridCell, GridCellHash> & came_from,
  const GridCell & start,
  const GridCell & goal) const
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

std::vector<GridCell> AStar::findPath(const GridCell & start, const GridCell & goal)
{
  std::vector<GridCell> empty_path;

  // Start / Goal 이 real obstacle 위면 바로 실패
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

  // Closed set
  std::unordered_map<GridCell, bool, GridCellHash> closed_set;

  // g_score
  std::unordered_map<GridCell, double, GridCellHash> g_score;

  // Parent (for path reconstruction)
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

  while (!open_set.empty()) {
    Node current = open_set.top();
    open_set.pop();

    // Goal 도달
    if (current.cell == goal) {
      return reconstructPath(came_from, start, goal);
    }

    // 이미 처리한 노드면 스킵
    if (closed_set[current.cell]) {
      continue;
    }

    closed_set[current.cell] = true;

    // 이웃 탐색
    std::vector<GridCell> neighbors = getNeighbors(current.cell);

    for (const auto & neighbor : neighbors) {
      if (closed_set[neighbor]) {
        continue;
      }

      double dx = static_cast<double>(neighbor.x - current.cell.x);
      double dy = static_cast<double>(neighbor.y - current.cell.y);
      double movement_cost = std::sqrt(dx * dx + dy * dy);

      // ---- 거리 기반 penalty 계산 ----
      double penalty = 0.0;
      double dist_cells = std::numeric_limits<double>::infinity();

      if (!distance_map_.empty()) {
        dist_cells = distance_map_[neighbor.y][neighbor.x];
      }

      if (std::isfinite(dist_cells)) {
        // 장애물에서 max_influence_dist_cells_ 셀 이내만 penalty 적용
        if (dist_cells < max_influence_dist_cells_) {
          // dist = 0일 때 max_penalty_,
          // dist = max_influence_dist_cells_ 일 때 0 이 되도록 선형 스케일
          double w = (max_influence_dist_cells_ - dist_cells) / max_influence_dist_cells_;
          if (w < 0.0) {
            w = 0.0;
          }
          penalty = max_penalty_ * w;
        }
      }

      double tentative_g = current.g_cost + movement_cost + penalty;

      auto it = g_score.find(neighbor);
      if (it == g_score.end() || tentative_g < it->second) {
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
