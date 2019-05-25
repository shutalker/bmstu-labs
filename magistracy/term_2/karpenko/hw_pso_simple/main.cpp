#include <fstream>
#include <iostream>
#include "pso_solver.h"

double VectorNorm(const std::vector<double> &vec) {
  double norm = 0.0;

  for (int i = 0; i < vec.size(); ++i)
    norm += vec[i] * vec[i];

  return sqrt(norm);
}

FitnessFunction sphericalFunction = [] (const std::vector<double> &pos) -> double {
  double result = 0.0;

  for (int i = 0; i < pos.size(); ++i)
    result += pos[i] * pos[i];

  return result;
};

int main() {
  const int MULTISTART = 100;
  const double GLOBAL_OPTIMA_VALUE_EPS = 1e-3;
  const double GLOBAL_OPTIMA_POSITION_EPS = 1e-2;

  int globalOptimaValueSatisfied = 0;
  int globalOptimaPositionSatisfied = 0;

  PSOSolver psoSolver;

  for (int iStart = 0; iStart < MULTISTART; ++iStart) {
    psoSolver.InitSolver(50, 20, sphericalFunction);
    std::cout << psoSolver.Run() << std::endl;

    if (psoSolver.GetBestFitnessValue() < GLOBAL_OPTIMA_VALUE_EPS)
      globalOptimaValueSatisfied += 1;

    if (VectorNorm(std::move(psoSolver.GetBestPosition())) < GLOBAL_OPTIMA_POSITION_EPS)
      globalOptimaPositionSatisfied += 1;
  }

  double globalOptimaValueProbability = (double)(globalOptimaValueSatisfied) / MULTISTART;
  double globalOptimaPositionProbability = (double)(globalOptimaPositionSatisfied) / MULTISTART;

  std::cout << "p1 = " << globalOptimaValueProbability << std::endl;
  std::cout << "p2 = " << globalOptimaPositionProbability << std::endl;

  return 0;
}
