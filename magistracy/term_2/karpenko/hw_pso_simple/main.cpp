#include <fstream>
#include <iostream>
#include "pso_solver.h"

FitnessFunction sphericalFunction = [] (const std::vector<double> &pos) -> double {
  double result = 0.0;

  for (int i = 0; i < pos.size(); ++i)
    result += pos[i] * pos[i];

  return result;
};

int main() {
  PSOSolver psoSolver;
  psoSolver.InitSolver(50, 16, sphericalFunction);
  std::ofstream results("res.txt");
  std::cout << psoSolver.Run(results) << std::endl;
  return 0;
}
