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

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
class PSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, 42>
    : public BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, 42> {
 public:
  void DoStop() {
    bool &stop = BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, 42>::stop;
    std::cout << "FUCK TEMPLATES: " << stop << std::endl;
  }
};

int main() {
  PSOSolver<2, SphericalFunction> solver1;
  solver1.DoStop();

  PSOSolver<2, SphericalFunction, 42> solver2;
  solver2.DoStop();

  return 0;
}
