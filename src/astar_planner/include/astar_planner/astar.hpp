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

  // 맵 해상도 [m/셀] 설정 (OccupancyGrid.info.resolution을 여기로 전달)
  void setResolution(double resolution_m);

  // start, goal 은 grid 좌표
  std::vector<GridCell> findPath(const GridCell & start, const GridCell & goal);

  // 필요하면 외부에서 penalty 범위[m], 크기 튜닝도 가능하게 열어둠
  void setInfluenceRangeMeters(double range_m) { influence_range_m_ = range_m; }
  void setMaxPenalty(double max_penalty) { max_penalty_ = max_penalty; }

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

  // penalty 관련 파라미터
  double max_penalty_;        // 장애물 바로 옆에서의 최대 penalty
  double influence_range_m_;  // 장애물에서 몇 m까지 penalty 줄지 (resolution과 무관)
  double resolution_m_;       // 맵 해상도 [m/셀]

  // distance_map_[y][x] = 가장 가까운 "real obstacle(값=2)" 까지의 거리(셀 단위)
  std::vector<std::vector<double>> distance_map_;


};

}  // namespace astar_planner

#endif  // ASTAR_PLANNER__ASTAR_HPP_
