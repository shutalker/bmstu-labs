#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <fstream>
#include <random>

#include "utils.h"
#include "rect_computing_grid.h"

static const int NODES_PER_AXIS = 256;

static const int M = 100; // dimension of approximated vector in simulation
static const int L = 8;   // length (in bytes) of one float-point number in simulation
static const double T = 1e-8; // time of function computation in simulation
static const double T_S = 5e-5; // latency between two processors before communication
static const double T_C = (1.0 / 80.0) * 1e-6; // communication time between two processors
static const double C_F_MAX = 1e5;

static const int DIM = 2;  // dimension of problem space

static const double LIN_CONSTRAINT_A = 0.6;
static const double LIN_CONSTRAINT_B = -0.2;

int GetNetworkDiameter(int processors) {
  return ceil(2 * sqrt(processors) - 1);
}

double GetAcceleration(const std::vector<NodeStat> &stats,
    double computationComplexity) {
  int networkDiameter = GetNetworkDiameter(stats.size());
  int nodesToCompute = 0;

  for (const auto &stat: stats)
    nodesToCompute += stat.nodesComputable;

  int nodesToComputePerProcessor = ceil(nodesToCompute / stats.size());
  double solutionTimeParallel = 2 * T_S + (DIM + M) * nodesToComputePerProcessor \
      * L * networkDiameter * T_C + T * nodesToComputePerProcessor * computationComplexity;
  double solutionTimeSequential = T * nodesToCompute * computationComplexity;

  return solutionTimeSequential / solutionTimeParallel;
}

double GetAverage(const std::vector<double> &vec) {
  double avg = 0.0;

  for (int i = 0; i < vec.size(); ++i)
    avg += vec[i];

  return avg / vec.size();
}

double GetDispersion(const std::vector<double> &vec, double avg) {
  double disp = 0.0;

  for (int i = 0; i < vec.size(); ++i)
    disp += pow((vec[i] - avg), 2);

  return (vec.size() > 1) ? disp / (vec.size() - 1) : 0.0;
}

int main() {
  std::function<bool(double, double)> linearConstraint = [] (double x, double y) -> bool {
    return (y - LIN_CONSTRAINT_A * x - LIN_CONSTRAINT_B) >= 0.0;
  };

  RectComputingGrid grid(0, 0, 1, 1);
  grid.ApplyNodeGrid(NODES_PER_AXIS, NODES_PER_AXIS);
  grid.AddLinearConstraint(linearConstraint);

  const std::vector<int> N_REPEATS = {1, 300};
  const std::vector<int> N_PROCESSORS = {2, 4, 8, 16, 32, 64, 128, 256};
  const std::vector<double> C_F_MAX = {8e6, 8e3};
  std::random_device randomDevice;
  std::mt19937 randomGenerator(randomDevice());
  std::ofstream results("res.out");

  for (const auto &nProc: N_PROCESSORS) {
    std::cout << "main --> processors = " << nProc << std::endl;
    std::vector<NodeStat> stats = std::move(grid.GetNodeStat(nProc));

    for (const auto &cfMax: C_F_MAX) {
      std::cout << "  main --> cf_max = " << cfMax << std::endl;
      std::uniform_real_distribution<> computationComlexityDist(0.0, cfMax);

      for (const auto &r: N_REPEATS) {
        std::cout << "    main --> repeats = " << r << std::endl;
        std::vector<double> accelerations(r, 0.0);
        double accelAverage = 0.0;
        double accelDispersion = 0.0;
        double accelDeviation = 0.0;

        if (r == 1) {
          accelAverage = GetAcceleration(stats, cfMax);
        } else {
          for (int iRepeat = 0; iRepeat < r; ++iRepeat) {
            accelerations[iRepeat] = GetAcceleration(stats, computationComlexityDist(randomGenerator));
          }

          accelAverage = GetAverage(accelerations);
          accelDispersion = GetDispersion(accelerations, accelAverage);
          accelDeviation = sqrt(accelDispersion);
        }

        results << std::setprecision(15)  << nProc << " " <<  cfMax << " "
            << r << " " << accelAverage << " " << accelDispersion << " "
            << accelDeviation << std::endl;
      }
    }

    results << std::endl << std::endl;
  }

  return 0;
}
