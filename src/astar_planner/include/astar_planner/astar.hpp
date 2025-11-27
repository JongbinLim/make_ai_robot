#ifndef ASTAR_PLANNER__ASTAR_HPP_
#define ASTAR_PLANNER__ASTAR_HPP_

#include <vector>
#include <unordered_map>

namespace astar_planner
{

struct GridCell
{
  int x;
  int y;

  bool operator==(const GridCell & other) const noexcept
  {
    return x == other.x && y == other.y;
  }
};

struct GridCellHash
{
  std::size_t operator()(const GridCell & c) const noexcept
  {
    // 간단한 hash 조합
    return std::hash<int>()(c.x) ^ (std::hash<int>()(c.y) << 1);
  }
};

struct Node
{
  GridCell cell;
  double g_cost;
  double h_cost;
  double f_cost;
  GridCell parent;

  bool operator>(const Node & other) const noexcept
  {
    return f_cost > other.f_cost;
  }
};

class AStar
{
public:
  AStar();
  ~AStar();

  // 0/1/2 맵 설정 (0: free, 1: margin, 2: real obstacle)
  void setMap(const std::vector<std::vector<int>> & map);

  // start, goal 은 grid 좌표
  std::vector<GridCell> findPath(const GridCell & start, const GridCell & goal);

private:
  double calculateHeuristic(const GridCell & a, const GridCell & b) const;
  bool isValid(const GridCell & cell) const;
  std::vector<GridCell> getNeighbors(const GridCell & cell) const;

  std::vector<GridCell> reconstructPath(
    const std::unordered_map<GridCell, GridCell, GridCellHash> & came_from,
    const GridCell & start,
    const GridCell & goal) const;

  // 원본 맵 (0: free, 1: margin, 2: obstacle)
  std::vector<std::vector<int>> map_;
  int map_width_;
  int map_height_;

  // distance_map_[y][x] = 가장 가까운 "real obstacle(값=2)" 까지의 거리(셀 단위)
  std::vector<std::vector<double>> distance_map_;

  // 거리 기반 soft cost 파라미터
  //   - 장애물 바로 옆(거리=0)에서의 penalty [cost]
  //   - 몇 셀 거리까지 penalty를 줄 것인지 (그 이상은 0)
  double max_penalty_;              // 예: 20.0
  double max_influence_dist_cells_; // 예: 8.0 셀

};

}  // namespace astar_planner

#endif  // ASTAR_PLANNER__ASTAR_HPP_
