#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "pso_solver.h"

std::random_device PSOSolver::randomDevice;
std::mt19937 PSOSolver::randomGenerator(randomDevice());

void PSOSolver::InitSolver(int swarmSize, int dimension, const FitnessFunction &func) {
  psoInertia = PSO_B_I;
  psoCognitial = PSO_B_C;
  psoSocial = PSO_B_S;
  searchSpaceDimension = dimension;
  fitnessFunction = func;

  particles.resize(swarmSize);
  SwarmRandomInit();
}

bool PSOSolver::StagnationStopCriteria() {
  static const double DELTA_FITNESS = 1e-5;
  static const int DELTA_ITERATIONS = 30;

  double d = fabs(globalBestFitnessValue - prevGlobalBestFitnessValue);

  if (d > DELTA_FITNESS) {
    stagnationIterations = 0;
    return false;
  }

  if (++stagnationIterations < DELTA_ITERATIONS)
    return false;

  return true;
}

bool PatricleFitnessComparator(const PSOParticle &p1, const PSOParticle &p2) {
  return p1.bestFitnessValue < p2.bestFitnessValue;
}

bool PSOSolver::Run(std::ostream *resultOutput) {
  static const int MAX_ITERATIONS = 10000;
  int iterations = 0;
  stagnationIterations = 0;
  prevGlobalBestFitnessValue = 0.0;

  while (!StagnationStopCriteria() && iterations < MAX_ITERATIONS) {
    if (resultOutput) {
      *resultOutput << iterations << " " << std::setprecision(15) << globalBestFitnessValue
          << std::endl;
    }

    prevGlobalBestFitnessValue = globalBestFitnessValue;
    int iBestParticle = 0;
    double currBestFitnessValue = globalBestFitnessValue;
    double scaleCoeff = ScaleCoefficient(iterations++);
    // на сферической функции масштабирование ухудшает сходимость
    psoInertia = PSO_B_I; // * scaleCoeff; // не нужно, сходимость хуже !!!
    psoCognitial = PSO_B_C; // * scaleCoeff;
    psoSocial = PSO_B_S; // * scaleCoeff;

    std::uniform_real_distribution<> cognitiveMultiplierDistribution(0.0, psoCognitial);
    std::uniform_real_distribution<> socialMultiplierDistribution(0.0, psoSocial);

    std::sort(particles.begin(), particles.end(), PatricleFitnessComparator);

    // ПОДГОНЧИК
    // смещаем лучшего агента в сторону пяти лучших агентов
    // for (int iDim = 0; iDim < searchSpaceDimension; ++iDim) {
    //   double coord = 0.0;

    //   for (int iBest = 1; iBest < 6; ++iBest)
    //     coord += particles[iBest].position[iDim];

    //   particles[0].position[iDim] = coord / 5;
    // }

    // double fitnessValue = fitnessFunction(particles[0].position);

    // if (fitnessValue < particles[0].bestFitnessValue) {
    //   particles[0].bestFitnessValue = fitnessValue;
    //   particles[0].bestPosition = particles[0].position;
    // }

    // if (fitnessValue < currBestFitnessValue) {
    //   currBestFitnessValue = fitnessValue;
    // }
    // КОНЕЦ ПОДГОНЧИКА

    for (int iParticle = 1; iParticle < particles.size(); ++iParticle) {
      PSOParticle &particle = particles[iParticle];

      double cognitiveVectorComponent;
      double socialVectorComponent;

      for (int iDim = 0; iDim < searchSpaceDimension; ++iDim) {
        cognitiveVectorComponent = particle.bestPosition[iDim] - particle.position[iDim];
        socialVectorComponent = globalBestPosition[iDim] - particle.position[iDim];

        particle.velocity[iDim] = psoInertia * particle.velocity[iDim]
            + cognitiveMultiplierDistribution(randomGenerator) * cognitiveVectorComponent
            + socialMultiplierDistribution(randomGenerator) * socialVectorComponent;

        particle.position[iDim] += particle.velocity[iDim];
      } // for particle dimension

      double currFitnessValue = fitnessFunction(particle.position);

      if (currFitnessValue < particle.bestFitnessValue) {
        particle.bestFitnessValue = currFitnessValue;
        particle.bestPosition = particle.position;
      }

      if (currFitnessValue < currBestFitnessValue) {
        currBestFitnessValue = currFitnessValue;
        iBestParticle = iParticle;
      }
    } // for particles

    globalBestFitnessValue = currBestFitnessValue;
    globalBestPosition = particles[iBestParticle].position;
  } // while iterations

  return iterations < MAX_ITERATIONS;
}

void PSOSolver::SwarmRandomInit() {
  static const double INIT_POSITION_INF = -2.5;
  static const double INIT_POSITION_SUP = 2.5;
  static const double INIT_VELOCITY_INF = 0.0;
  static const double INIT_VELOCITY_SUP = 1.0;

  std::uniform_real_distribution<> positionDistribution(INIT_POSITION_INF, INIT_POSITION_SUP);
  std::uniform_real_distribution<> velocityDistribution(INIT_VELOCITY_INF, INIT_VELOCITY_SUP);
  globalBestFitnessValue = std::numeric_limits<double>::max();
  int iBestParticle = 0;

  for (int iParticle = 0; iParticle < particles.size(); ++iParticle) {
    PSOParticle &particle = particles[iParticle];
    particle.position.resize(searchSpaceDimension);
    particle.bestPosition.resize(searchSpaceDimension);
    particle.velocity.resize(searchSpaceDimension);

    for (int iDim = 0; iDim < searchSpaceDimension; ++iDim) {
      particle.position[iDim] = particle.bestPosition[iDim] = positionDistribution(randomGenerator);
      particle.velocity[iDim] = velocityDistribution(randomGenerator);
    }

    particle.bestFitnessValue = fitnessFunction(particle.bestPosition);

    if (particle.bestFitnessValue < globalBestFitnessValue)
      iBestParticle = iParticle;
  }

  globalBestFitnessValue = particles[iBestParticle].bestFitnessValue;
  globalBestPosition = particles[iBestParticle].bestPosition;
}

double PSOSolver::ScaleCoefficient(double x) {
  return (2.0 / (1.0 + exp(0.002 * x)));
}
