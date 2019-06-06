#include "gbmo_solver.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <cmath>
#include <exception>

int main() {
  GBMOSolver<> solver;
  int populationSize = 50;
  double initialTemperature = 100;

  auto RastrigrinFunction = [](const double *vec, int vecSize) -> double {
    double out = 0.0;

    for (int i = 0; i < vecSize; ++i)
      out += (vec[i] * vec[i] - 10.0 * cos(2 * M_PI * vec[i]) + 10.0);

    return out;
  };

  auto SphericalFunction = [](const double *vec, int vecSize) -> double {
    double out = 0.0;

    for (int i = 0; i < vecSize; ++i)
      out += (vec[i] * vec[i]);

    return out;
  };

  if (!solver.Init(populationSize, initialTemperature, SphericalFunction)) {
    std::cout << "failed to initialize GBMO solver" << std::endl;
    return 1;
  }

  try {
    std::ofstream resultsOutput("res.txt");
    solver.Run(resultsOutput);
  } catch (std::exception &e) {
    std::cerr << "main -->exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "main --> unknown exception" << std::endl;
  }

  return 0;
}
