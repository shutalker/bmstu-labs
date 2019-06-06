#include <algorithm>
#include <iostream>
#include "squareword_greedy_solver.h"

int main() {
  const int N_AGENTS = 20;
  const int MULTISTART = 100;
  std::vector<SquarewordGreedySolver> solvers(N_AGENTS);

  for (int iSolver = 0; iSolver < solvers.size(); ++iSolver) {
    if (!solvers[iSolver].InitFromFile("field.conf")) {
      std::cout << "failed to initialize solver " << iSolver
          << " from file" << std::endl;
      return 1;
    }
  }

  int foundSolutions = 0;
  SquarewordGreedySolver::GameField solution;

  for (int iStart = 0; iStart < MULTISTART; ++iStart) {
    bool solutionFound = false;
    bool isFirstSearch = true;

    for (int iRetry = 0; iRetry < solvers.size(); ++iRetry) {
      std::vector<SquarewordGreedySolver::GameField> bestFields;

      for (int iSolver = 0; iSolver < solvers.size(); ++iSolver) {
        solvers[iSolver].DumpCurrentGameField(std::cout);
        std::cout << "alphabet: ";
        solvers[iSolver].DumpAlphabet(std::cout);
        std::cout << "non-fixed symbols:" << std::endl;
        solvers[iSolver].DumpNonFixedSymbols(std::cout);

        std::vector<SquarewordGreedySolver::GameField> terminalFields;

        if (solvers[iSolver].RunGreedySearch(isFirstSearch, &terminalFields)) {
          solution = terminalFields[0];
          solutionFound = true;
          break;
        }

        if (terminalFields.size() < solvers.size()) {
          std::cout << "it's not possible to continue search: "
              << "terminalFields.size() = " << terminalFields.size()
              << " < solvers.size() = " << solvers.size() << std::endl;

          return 1;
        }

        bestFields.insert(bestFields.end(), terminalFields.begin(),
            terminalFields.begin() + solvers.size());
      }

      if (solutionFound)
        break;

      if (bestFields.size() < solvers.size()) {
        std::cout << "it's not possible to continue search: "
            << "bestFields.size() = " << bestFields.size()
            << " < solvers.size() = " << solvers.size() << std::endl;

        return 1;
      }

      isFirstSearch = false;
      std::sort(bestFields.begin(), bestFields.end(),
          SquarewordGreedySolver::GameField::DuplicatedSymdolsComparator);

      for (int iSolver = 0; iSolver < solvers.size(); ++iSolver)
        solvers[iSolver].InitFromGameField(bestFields[iSolver]);
    }

    if (!solutionFound)
      std::cout << "no solution was found" << std::endl;
    else
      foundSolutions += 1;
  }

  std::cout << "Solution found:" << std::endl;
  solution.Dump(std::cout);
  double p = (double)(foundSolutions) / MULTISTART;
  std::cout << "solution detecting probability: " << p << std::endl;
  return 0;
}
