#ifndef _HW_PSO_PSO_SOLVER_H_
#define _HW_PSO_PSO_SOLVER_H_

#include "pso_particle.h"

#include <functional>
#include <ostream>
#include <vector>

static const int PSO_STOP_STAGNATION = 0;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
class PSOSolver;

// PSO solver based on clique neighbourhood topology
// TODO: use topology as parameter of solver
template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
class BasicPSOSolver {
 public:
  bool Init(int particlesCount);
  void Run(std::ostream &resultOutput);

 private:
  friend class PSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>;

  double prevBestFitnessValue = 0;
  double bestFitnessValue;
  std::vector<PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>> particles;

  std::function<bool()> stopCriteria = [this] { return true; };
};

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA = PSO_STOP_STAGNATION>
class PSOSolver: public BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA> {
 public:
  PSOSolver();

 private:
  int iterationCounter = 0;
  const int DELTA_ITERATIONS = 30;
  const double DELTA_FITNESS_FUNCTION = 1e-5;
};

#include "pso_solver.hpp"

#endif // _HW_PSO_PSO_SOLVER_H_
