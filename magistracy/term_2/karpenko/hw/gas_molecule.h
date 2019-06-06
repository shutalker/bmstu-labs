#ifndef _HW_GAS_MOLECULE_H_
#define _HW_GAS_MOLECULE_H_

#include <random>

int Signum(double num) {
  return (num < 0) ? -1 : 1;
}

template <int SEARCH_SPACE_DIMENSION = 2>
class GasMolecule {
 public:
  GasMolecule();

  void UpdateBrownianMotion(double temperature);
  void UpdateTurbulentMotion();
  void UpdateMass(double currFitness, double bestFitness, double worstFitness);
  void UpdateVelocityVector(const double *prevPosition);

  const double * GetPosition() const { return position; }
  const double * GetVelocity() const { return velocity; }
  double GetMass() const { return mass; }

 private:
  double mass;
  double position[SEARCH_SPACE_DIMENSION];
  double velocity[SEARCH_SPACE_DIMENSION];

  // Boltzmann's constant
  static constexpr double BOLTZMANN = 1.380649e-23;

  // turbulent rotational motion (TRM) parameters according to GBMO algorithm
  static constexpr double TRM_A = 0.5;
  static constexpr double TRM_B = 0.2;

  // random generators for molecules initial velocity and displacement
  static constexpr double DISPLACEMENT_INF = -2.5;
  static constexpr double DISPLACEMENT_SUP = 2.5;

  static constexpr double VELOCITY_INF = 0.0;
  static constexpr double VELOCITY_SUP = (double)(SEARCH_SPACE_DIMENSION) / 2;

  static constexpr double MASS_INIT = 1;
  static constexpr double MASS_EPS = 1e-21;

  static std::random_device randomDevice;
  static std::mt19937 randomGenerator;

  static std::uniform_real_distribution<> displacementDistribution;
  static std::uniform_real_distribution<> velocityDistribution;

  static double CircleMap(double x) {
    return std::fmod(x + TRM_B - (TRM_A / 2 * M_PI) * sin(2 * M_PI * x), 1.0);
  }
};

template <int SEARCH_SPACE_DIMENSION>
std::random_device GasMolecule<SEARCH_SPACE_DIMENSION>::randomDevice;

template <int SEARCH_SPACE_DIMENSION>
std::mt19937 GasMolecule<SEARCH_SPACE_DIMENSION>::randomGenerator(randomDevice());

template <int SEARCH_SPACE_DIMENSION>
std::uniform_real_distribution<> GasMolecule<SEARCH_SPACE_DIMENSION>::displacementDistribution(
    DISPLACEMENT_INF, DISPLACEMENT_SUP);

template <int SEARCH_SPACE_DIMENSION>
std::uniform_real_distribution<> GasMolecule<SEARCH_SPACE_DIMENSION>::velocityDistribution(
    VELOCITY_INF, VELOCITY_SUP);

#include "gas_molecule.hpp"

#endif // _HW_GAS_MOLECULE_H_
