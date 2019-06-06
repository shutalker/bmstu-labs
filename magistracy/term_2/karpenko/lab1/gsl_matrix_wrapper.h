#ifndef _LAB1_GSL_MATRIX_WRAPPER_H_
#define _LAB1_GSL_MATRIX_WRAPPER_H_

#include <gsl/gsl_matrix.h>
#include <cstring>
#include <iomanip>
#include <iostream>

class GslSquareMatrixWrapper {
 public:
  GslSquareMatrixWrapper(int dimension, double *data = nullptr) {
    matrix = gsl_matrix_alloc(dimension, dimension);

    if (data)
      std::memcpy(matrix->data, data, dimension * dimension * sizeof(double));
  }

  ~GslSquareMatrixWrapper() {
    if (matrix)
      gsl_matrix_free(matrix);
  }

  gsl_matrix * const GetMatrix() const { return matrix; }
  bool Dump(std::ostream &output) const {
    if (!matrix)
      return false;

    for (int iRow = 0; iRow < matrix->size1; ++iRow) {
      for (int iCol = 0; iCol < matrix->size2; ++iCol) {
        int iCell = iCol + iRow * matrix->size2;
        output << std::setw(2) << std::setfill(' ') << gsl_matrix_get(matrix, iRow, iCol) << " ";
      }

      output << std::endl;
    }

    return true;
  }

 private:
  gsl_matrix *matrix = nullptr;
};

#endif //_LAB1_GSL_MATRIX_WRAPPER_H_
