#include <iostream>
#include <fstream>
#include "pso_particle.h"
#include "pso_solver.h"

struct SphericalFunction {
  double operator()(const double *x, int vecSize) {
    double result = 0.0;

    for (int i = 0; i < vecSize; ++i)
      result += x[i] * x[i];

    return result;
  }
};

int main() {
  std::ofstream results("res.txt");
  PSOSolver<4, SphericalFunction> solver;
  solver.Init(50);
  solver.Run(results);

  return 0;
}
