#include "gauss_solver.h"
#include <iostream>
#include <algorithm>
#include <cmath>

GaussSolver::GaussSolver(const ExtendedMatrix &mtx)
    : matrix(mtx) {
}

void GaussSolver::Clear() {
    matrix.Clear();
    solution.reset();
}

void GaussSolver::Init(const ExtendedMatrix &mtx) {
    Clear();
    matrix = mtx;
}

bool GaussSolver::IsEqual(double x, double y) {
    return fabs(x - y) <= std::numeric_limits<double>::epsilon();
}

bool GaussSolver::IsZero(double val) {
    return IsEqual(val, 0.0);
}

bool GaussSolver::SwapRow(int iRowCol) {
    Vector &a = *matrix.src;
    int iTgt = iRowCol + 1;

    for (; iTgt < matrix.rows; ++iTgt) {
        if (!IsZero(a[iRowCol + matrix.cols * iTgt]))
            break;
    }

    if (!(iTgt < matrix.rows))
        return false;

    auto itStartSrc = a.begin() + iRowCol * matrix.cols;
    auto itEndSrc = itStartSrc + matrix.cols;
    auto itStartDest = a.begin() + iTgt * matrix.cols;

    std::swap_ranges(itStartSrc, itEndSrc, itStartDest);
    transmutations[iRowCol] = iTgt;
    return true;
}

double GaussSolver::ExpressFreeMember(int iRowCol) {
    const Vector &a = *matrix.src;
    Vector &x = *solution;
    double freeMember = a[(matrix.cols - 1) + matrix.cols * iRowCol];

    for (int iCol = iRowCol + 1; iCol < matrix.cols - 1; ++iCol)
        freeMember -= x[iCol] * a[iCol + matrix.cols * iRowCol];

    return freeMember;
}

void GaussSolver::ForwardTraverse() {
    Vector &a = *matrix.src;
    int iPrevDiag = 0;
    int iPrevRowCol = 0;

    for (int iCurrRow = 1; iCurrRow < matrix.rows; ++iCurrRow) {
        DumpMatrix();
        iPrevRowCol = iCurrRow - 1;
        iPrevDiag = iPrevRowCol + matrix.cols * iPrevRowCol;
        std::cout << iPrevRowCol << " " << iPrevDiag << std::endl << std::endl;

        if (IsZero(a[iPrevDiag]) && !SwapRow(iPrevRowCol))
            return;

        for (int iRow = iCurrRow; iRow < matrix.rows; ++iRow) {
            double k = a[iPrevRowCol + matrix.cols * iRow] / a[iPrevDiag];

            for (int iCol = 0; iCol < matrix.cols; ++iCol)
                a[iCol + matrix.cols * iRow] -= k * a[iCol + matrix.cols * iPrevRowCol];
        }
    }
}

void GaussSolver::BackwardTraverse() {
    Vector &a = *matrix.src;
    solution = std::make_shared<Vector>(matrix.rows);
    Vector &x = *solution;
    int iDiag = 0;
    int iFree = 0;
    int iVec  = 0;

    for (int iRow = matrix.rows - 1; iRow >= 0 ; --iRow) {
        iDiag = iRow + matrix.cols * iRow;
        iFree = (matrix.cols - 1)  + matrix.cols * iRow; // TODO: optimise

        iVec = (transmutations.find(iRow) == transmutations.end()) ? iRow
            : transmutations[iRow];

        if (IsZero(a[iDiag])) {
            if (!IsEqual(a[iDiag], ExpressFreeMember(iRow)))
                throw std::runtime_error("Equation system is not compatible!");

            x[iVec] = 0.0; // free variable;
            continue;
        }

        x[iVec] = a[iFree] / a[iDiag];

        for (size_t iCol = matrix.rows - 1; iCol > iRow; --iCol)
            x[iVec] -= x[iCol] * (a[iCol + matrix.cols * iRow] / a[iDiag]);
    }
}

void GaussSolver::DumpMatrix() {
    Vector &a = *matrix.src;

    for (size_t iRow = 0; iRow < matrix.rows; ++iRow) {
        for (size_t iCol = 0; iCol < matrix.cols; ++iCol) {
            std::cout << a[iCol + matrix.cols * iRow]
                << ((iCol < matrix.cols - 1) ? "\t" : "");
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void GaussSolver::Solve() {
        ForwardTraverse();
        DumpMatrix();
        BackwardTraverse();
}