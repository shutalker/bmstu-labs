#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cmath>

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
bool BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>::Init(int particlesCount) {
  if (particlesCount < 1)
    return false;

  particles.clear();
  particles.resize(particlesCount);

  return true;
}

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
void BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>::Run(
    std::ostream &resultOutput) {
  int iterations = 0;

  while(!stopCriteria()) {
    prevBestFitnessValue = bestFitnessValue;

    // clique neighbour topology
    std::sort(particles.begin(), particles.end(), [] (
        const PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION> &p1,
        const PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION> &p2) {
      return p1.GetBestFitnessValue() > p2.GetBestFitnessValue();
    });

    int iBestParticle = 0;
    int iSecondParticle = 1;
    const double *pos = particles[iBestParticle].GetBestPosition();
    bestFitnessValue = particles[iBestParticle].GetBestFitnessValue();
    resultOutput << iterations++ << " " << std::setprecision(15) \
      << std::fixed << bestFitnessValue << std::endl;

    for (int iParticle = iSecondParticle; iParticle < particles.size(); ++iParticle)
      particles[iParticle].UpdateBestNeighbourPosition(pos);

    if (particles.size() > 1)
      particles[iBestParticle].UpdateBestNeighbourPosition(particles[iSecondParticle].GetBestPosition());

    for (int iParticle = 0; iParticle < particles.size(); ++iParticle)
      particles[iParticle].UpdateSearchState();
  }
}

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION, int STOP_CRITERIA>
PSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>::PSOSolver() {
  BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION, STOP_CRITERIA>::stopCriteria = [this] {
    const double &bestFitnessValue = BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION,
         STOP_CRITERIA>::bestFitnessValue;
    const double &prevBestFitnessValue = BasicPSOSolver<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION,
         STOP_CRITERIA>::prevBestFitnessValue;

    if (std::fabs(prevBestFitnessValue - bestFitnessValue) < DELTA_FITNESS_FUNCTION) {
      if (++iterationCounter >= DELTA_ITERATIONS)
        return true;
    } else {
      iterationCounter = 0;
    }

    return false;
  };
}
