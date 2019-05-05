#include <fstream>
#include <functional>
#include <iomanip>
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
  const std::vector<double> C_F_MAX = {8e6, 8e3}; // computation complexity (number of operations per one computation)

  std::ofstream results("res.out");

  for (const auto &nProc: N_PROCESSORS) {
    std::cout << "main --> processors = " << nProc << std::endl;
    std::vector<NodeStat> stats = std::move(grid.GetNodeStat(nProc));

    std::vector<std::shared_ptr<MPCBAnlyzer>> analyzers;
    analyzers.emplace_back(new MPCBAnalyzerSpaceDecomposition());
    analyzers.emplace_back(new MPCBAnalyzerNodesDecomposition());

    for (const auto &cfMax: C_F_MAX) {
      std::cout << "  main --> cf_max =  " << cfMax << std::endl;
      results << std::setprecision(15) << nProc << " " << cfMax << " ";

      for (const auto &analyzer: analyzers) {
        AccelerationStat accelStat = std::move(analyzer->GetAccelerationStat(stats, cfMax));
        std::cout << "    main --> " << analyzer->GetName() <<  " acceleration = "
            << accelStat.accelleration << std::endl;
        results << std::setprecision(15) << accelStat.accelleration << " ";
      }

      results << std::endl;
    }

    results << std::endl << std::endl;
  }

  return 0;
}
