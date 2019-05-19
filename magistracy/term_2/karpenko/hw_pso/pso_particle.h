#ifndef _HW_PSO_PSO_PARTICLE_H_
#define _HW_PSO_PSO_PARTICLE_H_

#include <random>
#include <cstdint> // uint32_t typedef

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
class PSOParticle {
 public:
  PSOParticle(): id(idCounter++) { RandomInit(); }

  void UpdateSearchState();
  void UpdateBestNeighbourPosition(const double *pos);

  uint32_t GetId() const { return id; }
  double GetBestFitnessValue() const { return bestFitnessValue; }
  const double * GetBestPosition() const  { return bestPosition; }

 private:
  uint32_t id;
  bool isBestFitnessValueInitialized = false;
  double bestFitnessValue;

  double position[SEARCH_SPACE_DIMENSION];
  double bestPosition[SEARCH_SPACE_DIMENSION];
  double bestNeighbourPosition[SEARCH_SPACE_DIMENSION];

  double velocity[SEARCH_SPACE_DIMENSION];

  static constexpr double INIT_DISPLACEMENT_INF = -2.5;
  static constexpr double INIT_DISPLACEMENT_SUP = 2.5;

  static constexpr double INIT_VELOCITY_INF = 0.0;
  static constexpr double INIT_VELOCITY_SUP = 1.0;

  static constexpr double PSO_B_I = 0.72980; // inertia component coefficient
  static constexpr double PSO_B_C = 1.49618; // congnitive component coefficient
  static constexpr double PSO_B_S = 1.49618; // social component coefficient

  static uint32_t idCounter; // not thread-safe
  static FITNESS_FUNCTION fitnessFunction;
  static std::random_device randomDevice;
  static std::mt19937 randomGenerator; // not thread-safe
  // distributions are cheap so it can be implemented inside function members

  void RandomInit();
};

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
uint32_t PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::idCounter = 0;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
FITNESS_FUNCTION PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::fitnessFunction;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
std::random_device PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::randomDevice;

template <int SEARCH_SPACE_DIMENSION, class FITNESS_FUNCTION>
std::mt19937 PSOParticle<SEARCH_SPACE_DIMENSION, FITNESS_FUNCTION>::randomGenerator(randomDevice());

#include "pso_particle.hpp"

#endif // _HW_PSO_PSO_PARTICLE_H_
