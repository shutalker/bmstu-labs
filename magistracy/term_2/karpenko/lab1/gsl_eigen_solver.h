#ifndef _LAB1_GSL_EIGEN_SOLVER_H_
#define _LAB1_GSL_EIGEN_SOLVER_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <iostream>

class EigenSquareSymMatrixSolver {
 public:
  EigenSquareSymMatrixSolver(gsl_matrix *symMatrix, int dim)
      : dimension(dim), matrix(symMatrix) {
    workspace = gsl_eigen_symmv_alloc(dimension);
    eigenValues = gsl_vector_alloc(dimension);
    eigenVectors = gsl_matrix_alloc(dimension, dimension);
  }

  ~EigenSquareSymMatrixSolver() {
    gsl_eigen_symmv_free(workspace);
    gsl_vector_free(eigenValues);
    gsl_matrix_free(eigenVectors);
  }

  bool Solve() {
    if (gsl_eigen_symmv(matrix, eigenValues, eigenVectors, workspace) != 0) {
      std::cout << "EigenSquareSymMatrixSolver::Solve --> error in gsl_eigen_symmv"
          << std::endl;
      return false;
    }

    if (gsl_eigen_symmv_sort(eigenValues, eigenVectors, GSL_EIGEN_SORT_ABS_ASC) != 0) {
      std::cout << "EigenSquareSymMatrixSolver::Solve --> error in gsl_eigen_symmv_sort"
          << std::endl;
      return false;
    }

    return true;
  }

  gsl_vector_view GetEigenVector(int iVec) const noexcept(false) {
    if (iVec < 0 || iVec >= dimension) {
      throw std::runtime_error("EigenSquareSymMatrixSolver::GetEigenVector --> iVec out of range: "
          + std::to_string(iVec) + "; dimension = " + std::to_string(dimension));
    }

    gsl_vector_view vec = gsl_matrix_column(eigenVectors, iVec);

    if (!vec.vector.data)
      throw std::runtime_error("EigenSquareSymMatrixSolver::GetEigenVector --> vector is nullptr");

    return vec;
  }

 private:
  int dimension;
  gsl_matrix *matrix;
  gsl_eigen_symmv_workspace *workspace;
  gsl_vector *eigenValues;
  gsl_matrix *eigenVectors;
};

#endif //_LAB1_GSL_EIGEN_SOLVER_H_
