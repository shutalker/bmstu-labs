#include "gbmo_solver.h"
#include <cstring>
#include <iomanip>
#include <iostream>

template <int SEARCH_SPACE_DIMENSION>
bool GBMOSolver<SEARCH_SPACE_DIMENSION>::Init(int populationSize, double initialTemperature,
    const std::function<double(const double *, int)> &objectFunc) {
  Clear();

  if (initialTemperature <= 0.0) {
    std::cout << "GBMOSolver::Init --> invalid initialTemperature (should be > 0): "
        << initialTemperature << std::endl;
    return false;
  }

  if (populationSize <= 0) {
    std::cout << "GBMOSolver::Init --> negative populationSize (should be > 0): "
        << populationSize << std::endl;
    return false;
  }

  temperature = initialTemperature;
  fitnessValues.resize(populationSize);
  molecules.resize(populationSize);
  objectFunction = objectFunc;
  std::cout << "GBMOSolver::Init --> initial temperature = " << temperature << std::endl;
  std::cout << "GBMOSolver::Init --> population size = " << populationSize << std::endl;
  return true;
}

template <int SEARCH_SPACE_DIMENSION>
void GBMOSolver<SEARCH_SPACE_DIMENSION>::Run(std::ostream &output) {
  int iterationCounter = 0;
  double bestFitnessValue;
  double worstFitnessValue;
  double meanFitnessValue;
  bool isBestFitnessInitialized;
  bool isWorstFitnessInitialized;

  while (temperature > 0) {
    std::cout << "GBMOSolver::Run --> temperature = " << temperature << std::endl;
    isBestFitnessInitialized  = false;
    isWorstFitnessInitialized = false;
    meanFitnessValue = 0.0;

    for(int iMolecule = 0; iMolecule < molecules.size(); ++iMolecule) {
      auto &molecule = molecules[iMolecule];
      // std::cout << "GBMOSolver::Run --> molecule mass = " << molecule.GetMass() << std::endl;
      molecule.UpdateBrownianMotion(temperature);

      const double *p = molecule.GetPosition();
      std::cout << "GBMOSolver::Run --> molecule position: ";
      for (int j = 0; j < SEARCH_SPACE_DIMENSION; ++j) {
        std::cout << p[j] << " ";
      }

      std::cout << std::endl;

      // const double *v = molecule.GetVelocity();
      // std::cout << "GBMOSolver::Run --> molecule velocity: ";
      // for (int j = 0; j < SEARCH_SPACE_DIMENSION; ++j) {
      //   std::cout << v[j] << " ";
      // }

      // std::cout << std::endl;

      molecule.UpdateTurbulentMotion();
      double updatedFitnessValue = objectFunction(molecule.GetPosition(),
          SEARCH_SPACE_DIMENSION);

      p = molecule.GetPosition();
      std::cout << "GBMOSolver::Run --> molecule NEW position: ";
      for (int j = 0; j < SEARCH_SPACE_DIMENSION; ++j) {
        std::cout << p[j] << " ";
      }

      std::cout << std::endl;

      fitnessValues[iMolecule] = updatedFitnessValue;
      std::cout << "GBMOSolver::Run --> fitnessValue = " << fitnessValues[iMolecule] << std::endl;

      // update best fitness value
      if (!isBestFitnessInitialized || fitnessValues[iMolecule] < bestFitnessValue) {
        isBestFitnessInitialized = true;
        bestFitnessValue = fitnessValues[iMolecule];
      }

      // update worst fitness value
      if (!isWorstFitnessInitialized || fitnessValues[iMolecule] > worstFitnessValue) {
        isWorstFitnessInitialized = true;
        worstFitnessValue = fitnessValues[iMolecule];
      }

      // update mean fitness value
      meanFitnessValue = (meanFitnessValue * iMolecule + fitnessValues[iMolecule]) / (iMolecule + 1);
      std::cout << "GBMOSolver::Run --> iMolecule = " << iMolecule
          << "; currentFitnessValue = " << fitnessValues[iMolecule]
          << "; meanFitnessValue = " << meanFitnessValue << std::endl;
    }

    for(int iMolecule = 0; iMolecule < molecules.size(); ++iMolecule) {
      molecules[iMolecule].UpdateMass(fitnessValues[iMolecule], bestFitnessValue,
          worstFitnessValue);
      std::cout << "GBMOSolver::Run --> molecules[" << iMolecule << "].mass = "
          << molecules[iMolecule].GetMass() << std::endl;
    }

    std::cout << "GBMOSolver::Run --> bestFitnessValue = " << bestFitnessValue << std::endl;
    std::cout << "GBMOSolver::Run --> worstFitnessValue = " << worstFitnessValue << std::endl;
    std::cout << "GBMOSolver::Run --> meanFitnessValue = " << meanFitnessValue << std::endl;
    temperature -= 1.0 / meanFitnessValue;
    output << std::setprecision(15) << bestFitnessValue << '\t'
        << (iterationCounter++) << std::endl;

    std::string s;
    std::getline(std::cin, s);
  }
}

template <int SEARCH_SPACE_DIMENSION>
void GBMOSolver<SEARCH_SPACE_DIMENSION>::Clear() {
  fitnessValues.clear();
  molecules.clear();
  std::function<double(const double *, int)> t;
  objectFunction.swap(t);
}
