#include "game_field.h"
#include "game_solver.h"
#include <iostream>
#include <fstream>

int main() {
  try {
    GameField gf("field.conf.json");
    std::cout << "Initial field:" << std::endl;
    std::cout <<  gf.Dump() << std::endl;

    GameSolver solver(gf);

    if (solver.FindSolution("PAKETA")) {
      std::ofstream output("solution.txt", std::ios::binary | std::ios::out);
      solver.DumpSolution(output);
    }
  } catch (const std::exception &e) {
    std::cerr << "exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown exception" << std::endl;
    return 2;
  }

  return 0;
}
