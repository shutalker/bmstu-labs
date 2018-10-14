#include "GaussSolver/gauss_solver.h"
#include <iostream>

int main() {
    ExtendedMatrix m;

    m.src = std::make_shared<Vector>(Vector{
        1.0, 2.0, 4.00,
        2.0, 3.999, 7.998
    });

    m.rows = 2;
    m.cols = 3;

    GaussSolver gSolver(m);
    gSolver.Solve();
    gSolver.DumpMatrix();
    std::cout << std::endl;

    VectorPtr sol = gSolver.GetSolution();

    for (size_t i = 0; i < sol->size(); ++i)
        std::cout << (*sol)[i] << std::endl;

    return 0;
}