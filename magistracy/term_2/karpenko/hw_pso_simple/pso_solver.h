#ifndef _HW_PSO_SIMPLE_PSO_SOLVER_H_
#define _HW_PSO_SIMPLE_PSO_SOLVER_H_

#include <functional>
#include <ostream>
#include <random>
#include <vector>

typedef std::function<double(const std::vector<double> &)> FitnessFunction;

struct PSOParticle {
  double bestFitnessValue;
  std::vector<double> bestPosition;

  std::vector<double> position;
  std::vector<double> velocity;
};

class PSOSolver {
 public:
  void InitSolver(int swarmSize, int dimension, const FitnessFunction &func);
  bool Run(std::ostream *resultOutput = nullptr);
  double GetBestFitnessValue() const { return globalBestFitnessValue; }
  std::vector<double> GetBestPosition() const { return globalBestPosition; }

 private:
  int searchSpaceDimension;
  double globalBestFitnessValue;

  double psoInertia;
  double psoCognitial;
  double psoSocial;

  int stagnationIterations;
  double prevGlobalBestFitnessValue;

  std::vector<double> globalBestPosition;
  std::vector<PSOParticle> particles;
  FitnessFunction fitnessFunction;

  static std::random_device randomDevice;
  static std::mt19937 randomGenerator;

  static constexpr double PSO_B_I = 0.72980; // inertia component coefficient
  static constexpr double PSO_B_C = 1.49618; // congnitive component coefficient
  static constexpr double PSO_B_S = 1.49618; // social component coefficient

  void SwarmRandomInit();
  bool StagnationStopCriteria();
  double ScaleCoefficient(double x); // 2 / (1 + e^(0.003x)) - scale coefficient
};

#endif // _HW_PSO_SIMPLE_PSO_SOLVER_H_
