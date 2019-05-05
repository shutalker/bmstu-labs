#ifndef _LAB2_MPCB_ANALYZER_H_
#define _LAB2_MPCB_ANALYZER_H_

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "utils.h"

static const int M = 100; // dimension of approximated vector in simulation
static const int L = 8;   // length (in bytes) of one float-point number in simulation
static const double T = 1e-8; // time of function computation in simulation
static const double T_S = 5e-5; // latency between two processors before communication
static const double T_C = (1.0 / 80.0) * 1e-6; // communication time between two processors

static const int DIM = 2;  // dimension of problem space
static const int C_F = 100000; // computation complexity (number of operations per one computation)

// MultiProcessor Computation Balancing Analyzer base class
class MPCBAnlyzer {
 public:
  virtual ~MPCBAnlyzer() {}
  virtual AccelerationStat GetAccelerationStat(const std::vector<NodeStat> &stats) = 0;
  virtual std::string GetName() = 0;

  int GetNetworkDiameter(int processors) {
    return ceil(2 * sqrt(processors) - 1);
  }
};

class MPCBAnalyzerSpaceDecomposition: public MPCBAnlyzer {
 public:
  virtual AccelerationStat GetAccelerationStat(const std::vector<NodeStat> &stats) {
    double solutionTimeParallel = 0.0;
    int networkDiameter = GetNetworkDiameter(stats.size());
    int nodesToCompute = 0;

    for (const auto &stat: stats) {
      nodesToCompute += stat.nodesComputable;
      double t = 2 * T_S + (stat.nodesOverall * DIM + stat.nodesComputable * M) \
          * L * networkDiameter * T_C + T * stat.nodesComputable * C_F;

      if (t > solutionTimeParallel)
        solutionTimeParallel = t;
    }

    double solutionTimeSequential = T * nodesToCompute * C_F;
    AccelerationStat accelStat;
    accelStat.accelleration = solutionTimeSequential / solutionTimeParallel;

    return accelStat;
  }

  virtual std::string GetName() { return "SpaceDecomposition"; }
};

class MPCBAnalyzerNodesDecomposition: public MPCBAnlyzer {
 public:
  virtual AccelerationStat GetAccelerationStat(const std::vector<NodeStat> &stats) {
    int networkDiameter = GetNetworkDiameter(stats.size());
    int nodesToCompute = 0;

    for (const auto &stat: stats)
      nodesToCompute += stat.nodesComputable;

    int nodesToComputePerProcessor = ceil(nodesToCompute / stats.size());
    double solutionTimeParallel = 2 * T_S + (DIM + M) * nodesToComputePerProcessor \
        * L * networkDiameter * T_C + T * nodesToComputePerProcessor * C_F;
    double solutionTimeSequential = T * nodesToCompute * C_F;
    AccelerationStat accelStat;
    accelStat.accelleration = solutionTimeSequential / solutionTimeParallel;

    return accelStat;
  }

  virtual std::string GetName() { return "NodesDecomposition"; }
};

#endif // _LAB2_MPCB_ANALYZER_H_
