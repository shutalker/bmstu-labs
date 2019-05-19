#include <cstring>

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
void PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::RandomInit() {
  std::uniform_real_distribution<> displacementDistribution(INIT_DISPLACEMENT_INF,
    INIT_DISPLACEMENT_SUP);
  std::uniform_real_distribution<> velocityDistribution(INIT_VELOCITY_INF,
    INIT_VELOCITY_SUP);

  for (int iDim = 0; iDim < SEARCH_SPACE_DIMENSION; ++iDim) {
    position[iDim] = displacementDistribution(randomGenerator);
    velocity[iDim] = velocityDistribution(randomGenerator);
  }
}

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
void PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::UpdateSearchState() {
  std::uniform_real_distribution<> cognitiveMultiplierDistribution(0.0, PSO_B_C);
  std::uniform_real_distribution<> socialMultiplierDistribution(0.0, PSO_B_S);

  double cognitiveMultiplier = cognitiveMultiplierDistribution(randomGenerator);
  double socialMultiplier = socialMultiplierDistribution(randomGenerator);
  double cognitiveVectorComponent;
  double socialVectorComponent;

  for (int iDim = 0; iDim < SEARCH_SPACE_DIMENSION; ++iDim) {
    cognitiveVectorComponent = bestPosition[iDim] - position[iDim];
    socialVectorComponent = bestNeighbourPosition[iDim] - position[iDim];

    velocity[iDim] = PSO_B_I * velocity[iDim] + cognitiveMultiplier * cognitiveVectorComponent \
        + socialMultiplier * socialVectorComponent;

    position[iDim] += velocity[iDim];
  }

  double currFitnessValue = fitnessFunction(position, SEARCH_SPACE_DIMENSION);

  if (!isBestFitnessValueInitialized || currFitnessValue < bestFitnessValue) {
    isBestFitnessValueInitialized = true;
    bestFitnessValue = currFitnessValue;
    std::memcpy(bestPosition, position, sizeof(double) * SEARCH_SPACE_DIMENSION);
  }
}

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
void PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::UpdateBestNeighbourPosition(const double *pos) {
  std::memcpy(bestNeighbourPosition, pos, sizeof(double) * SEARCH_SPACE_DIMENSION);
}
