#include <functional>
#include <iostream>
#include <memory>

#include "rect_computing_grid.h"
#include "mpcb_analyzer.h"

static const int NODES_PER_AXIS = 256;

static const double LIN_CONSTRAINT_A = 0.6;
static const double LIN_CONSTRAINT_B = -0.2;

int main() {
  std::function<bool(double, double)> linearConstraint = [] (double x, double y) -> bool {
    return (y - LIN_CONSTRAINT_A * x - LIN_CONSTRAINT_B) >= 0.0;
  };

  RectComputingGrid grid(0, 0, 1, 1);
  grid.ApplyNodeGrid(NODES_PER_AXIS, NODES_PER_AXIS);
  grid.AddLinearConstraint(linearConstraint);

  const std::vector<int> N_PROCESSORS = {2, 4, 8, 16, 32, 64};

  for (const auto &nProc: N_PROCESSORS) {
    std::cout << "main --> processors = " << nProc << std::endl;
    std::vector<NodeStat> stats = std::move(grid.GetNodeStat(nProc));

    std::vector<std::shared_ptr<MPCBAnlyzer>> analyzers;
    analyzers.emplace_back(new MPCBAnalyzerSpaceDecomposition());
    analyzers.emplace_back(new MPCBAnalyzerNodesDecomposition());

    std::cout << "main --> acceleration for " << nProc << " processors:" << std::endl;

    for (const auto &analyzer: analyzers) {
      AccelerationStat accelStat = std::move(analyzer->GetAccelerationStat(stats));
      std::cout << "  " << analyzer->GetName() <<  " acceleration = " << accelStat.accelleration << std::endl;
    }
  }

  return 0;
}
