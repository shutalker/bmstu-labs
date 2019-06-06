#ifndef _HW_GBMO_SOLVER_H_
#define _HW_GBMO_SOLVER_H_

#include "gas_molecule.h"
#include <functional>
#include <ostream>
#include <vector>

template <int SEARCH_SPACE_DIMENSION = 2>
class GBMOSolver {
 public:
  bool Init(int populationSize, double initialTemperature,
    const std::function<double(const double *, int)> &objectFunc);

  void Run(std::ostream &output);
  void Clear();

 private:
  double temperature;

  std::vector<double> fitnessValues;
  std::vector<GasMolecule<SEARCH_SPACE_DIMENSION>> molecules;
  std::function<double(const double *, int)> objectFunction;
};

#include "gbmo_solver.hpp"

#endif // _HW_GBMO_SOLVER_H_
