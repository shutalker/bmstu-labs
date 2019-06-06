#include "rect_computing_grid.h"
#include <cmath>
#include <iostream>

RectComputingGrid::RectComputingGrid(double xInf, double yInf,
    double xSup, double ySup) {
  SetRect(xInf, yInf, xSup, ySup);
}

void RectComputingGrid::SetRect(double xInf, double yInf, double xSup, double ySup) {
    xRect0 = xInf;
    yRect0 = yInf;
    xRect1 = xSup;
    yRect1 = ySup;

    rectWidth = fabs(xRect1 - xRect0);
    rectHeight = fabs(yRect1 - yRect0);
    xDirection = Signum(xRect1 - xRect0);
    yDirection = Signum(yRect1 - yRect0);
}

bool RectComputingGrid::ApplyNodeGrid(int xNodes, int yNodes) {
  if (xNodes < 1)
    return false;

  if (yNodes < 1)
    return false;

  nodesX = xNodes;
  nodesY = yNodes;

  double xStart = (xNodes > 1) ? xRect0 : xRect0 + xDirection * (rectWidth / 2);
  double xStep  = (xNodes > 1) ? rectWidth / (xNodes - 1) : rectWidth;
  xStep *= xDirection;

  double yStart = (yNodes > 1) ? yRect0 : yRect0 + yDirection * (rectHeight / 2);
  double yStep  = (yNodes > 1) ? rectWidth / (yNodes - 1) : rectHeight;
  yStep *= yDirection;

  for (double x = xStart; fabs(x) <= fabs(xRect1); x += xStep) {
    for (double y = yStart; fabs(y) <= fabs(yRect1); y += yStep)
      nodes[x].emplace_back(y);
  }

  return true;
}

void RectComputingGrid::AddLinearConstraint(const LinearConstraint &constraint) {
  linearConstraints.emplace_back(constraint);
}

std::vector<NodeStat> RectComputingGrid::GetNodeStat(int subspaceCount) {
  if (subspaceCount < 1) {
    throw std::runtime_error("RectComputingGrid::GetNodeStat --> invalid value of "\
      "subspaceCount: " + std::to_string(subspaceCount));
  }

  std::vector<NodeStat> stats(subspaceCount);
  double xSubspaceStep = (rectWidth / subspaceCount) * xDirection;
  double xSubspaceStart = xRect0;

  for (int iSubspace = 0; iSubspace < subspaceCount; ++iSubspace) {
    double xSubspaceEnd = xSubspaceStart + xSubspaceStep;

    if (iSubspace == subspaceCount - 1)
      xSubspaceEnd += xSubspaceStep / 2; // perhaps it's better to use xNodeStep

    for (const auto &xLine: nodes) {
      // xLine.first --> x coordinate
      if (xLine.first < xSubspaceStart || xLine.first >= xSubspaceEnd)
        continue;

      for (const auto &y: xLine.second) {
        bool isNodeComputable = true;
        stats[iSubspace].nodesOverall += 1;

        for (const auto &constraint: linearConstraints)
          isNodeComputable &= constraint(xLine.first, y);

        if (isNodeComputable)
          stats[iSubspace].nodesComputable += 1;
      }
    }

    xSubspaceStart = xSubspaceEnd;
  }

  return stats;
}
