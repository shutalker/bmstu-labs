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

FitnessFunction rastriginFunction = [] (const std::vector<double> &pos) -> double {
  double result = 0.0;

  for (int i = 0; i < pos.size(); ++i)
    result += pos[i] * pos[i] - 10 * cos(2 * M_PI * pos[i]) + 10;

  return result;
};

int main() {
  const int MULTISTART = 100;
  const double GLOBAL_OPTIMA_VALUE_EPS = 1e-3;
  const double GLOBAL_OPTIMA_POSITION_EPS = 1e-2;

  int globalOptimaValueSatisfied = 0;
  int globalOptimaPositionSatisfied = 0;
  int swarmSize = 50;
  int dimension = 16;

  PSOSolver psoSolver;

  for (int iStart = 0; iStart < MULTISTART; ++iStart) {
    psoSolver.InitSolver(swarmSize, dimension, rastriginFunction);
    psoSolver.Run();

    std::cout << "iStart = " << (iStart + 1) << "; bestFitnessValue = "
        << psoSolver.GetBestFitnessValue() << std::endl;

    std::cout << "  ";

    for (const auto &p: psoSolver.GetBestPosition())
      std::cout << p << " ";

    std::cout << std::endl;

    if (psoSolver.GetBestFitnessValue() < GLOBAL_OPTIMA_VALUE_EPS)
      globalOptimaValueSatisfied += 1;

    if (VectorNorm(std::move(psoSolver.GetBestPosition())) < GLOBAL_OPTIMA_POSITION_EPS)
      globalOptimaPositionSatisfied += 1;
  }

  double globalOptimaValueProbability = (double)(globalOptimaValueSatisfied) / MULTISTART;
  double globalOptimaPositionProbability = (double)(globalOptimaPositionSatisfied) / MULTISTART;

  std::cout << "global optima localization probability: " << globalOptimaValueProbability << std::endl;
  std::cout << "global optima localization probability by variation vector: " << globalOptimaPositionProbability << std::endl;

  std::ofstream resultOutput("res.txt");
  psoSolver.InitSolver(swarmSize, dimension, rastriginFunction);
  psoSolver.Run(&resultOutput);

  return 0;
}
