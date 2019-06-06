#include <cmath>

template <int SEARCH_SPACE_DIMENSION>
GasMolecule<SEARCH_SPACE_DIMENSION>::GasMolecule() {

  for (int i = 0; i < SEARCH_SPACE_DIMENSION; ++i) {
    mass = MASS_INIT;
    position[i] = displacementDistribution(randomGenerator);
    velocity[i] = velocityDistribution(randomGenerator);
  }
}

template <int SEARCH_SPACE_DIMENSION>
void GasMolecule<SEARCH_SPACE_DIMENSION>::UpdateBrownianMotion(double temperature) {
  for (int i = 0; i < SEARCH_SPACE_DIMENSION; ++i) {
    velocity[i] += sqrt(3.0 * BOLTZMANN * temperature / mass);
    position[i] += velocity[i];
  }
}

template <int SEARCH_SPACE_DIMENSION>
void GasMolecule<SEARCH_SPACE_DIMENSION>::UpdateTurbulentMotion() {
  for (int i = 0; i < SEARCH_SPACE_DIMENSION; ++i)
    position[i] = CircleMap(position[i]);
}

template <int SEARCH_SPACE_DIMENSION>
void GasMolecule<SEARCH_SPACE_DIMENSION>::UpdateMass(double currFitness,
    double bestFitness, double worstFitness) {
  mass = (currFitness - worstFitness) / (bestFitness - worstFitness);

  if (std::isnan(mass))
    mass = MASS_EPS;
  else
    mass += MASS_EPS;
}
