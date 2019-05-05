#ifndef _LAB3_RECT_COMPUTING_GRID_H_
#define _LAB3_RECT_COMPUTING_GRID_H_

#include <functional>
#include <map>
#include <vector>

#include "utils.h"

class RectComputingGrid {
 public:
  typedef std::function<bool(double x, double y)> LinearConstraint;

  RectComputingGrid(double xInf, double yInf, double xSup, double ySup);
  void SetRect(double xInf, double yInf, double xSup, double ySup);
  bool ApplyNodeGrid(int xNodes, int yNodes);
  void AddLinearConstraint(const LinearConstraint &constraint);
  std::vector<NodeStat> GetNodeStat(int subspaceCount);

 private:
  RectComputingGrid() = delete;
  double xRect0, yRect0;
  double xRect1, yRect1;
  double rectWidth;
  double rectHeight;
  int xDirection, yDirection;
  int nodesX, nodesY;

  std::vector<LinearConstraint> linearConstraints;
  std::map<double, std::vector<double>> nodes; // x --> [y1, y2, ...]

  int Signum(double x) {
    return (x < 0) ? -1: 1;
  }
};

#endif // _LAB3_RECT_COMPUTING_GRID_H_
