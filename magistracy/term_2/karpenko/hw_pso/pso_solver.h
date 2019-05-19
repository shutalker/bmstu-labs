#ifndef _HW_PSO_PSO_SOLVER_H_
#define _HW_PSO_PSO_SOLVER_H_

#include "pso_particle.h"
#include <iostream>
#include <vector>

static const int PSO_STOP_STAGNATION = 0;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
class PSOSolver;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
class BasicPSOSolver {
 public:
  void DoStop() {
    std::cout << "[default] stop: " << stop << std::endl;
  }

 private:
  friend class PSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>;
  //std::vector<PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>> particles;
  bool stop = false;
};

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA = PSO_STOP_STAGNATION>
class PSOSolver: public BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA> {
 public:
  void DoStop() {
    bool &stop = BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>::stop;
    stop = true;

    std::cout << "[PSO_STOP_STAGNATION] stop: " << stop << std::endl;
  }
};

#endif // _HW_PSO_PSO_SOLVER_H_
