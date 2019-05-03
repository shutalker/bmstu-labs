#include "gsl_eigen_solver.h"
#include "gsl_matrix_wrapper.h"

#include <iostream>

double adjacencyMatrixDim = 15;
double adjacencyMatrix[] = {
//Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10Q11Q12Q13Q14Q15
  0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q1
  0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q2
  1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q3
  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q4
  0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q5
  0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // Q6
  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, // Q7
  0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // Q8
  0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, // Q9
  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, // Q10
  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, // Q11
  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, // Q12
  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, // Q13
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, // Q14
  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0  // Q15
};

bool InitFromAdjacencyMatrix(gsl_matrix *dest, gsl_matrix *src) {
  if (src->size1 != dest->size1) {
    std::cerr << "InitFromAdjacencyMatrix --> src->size1 != dest->size1" << std::endl;
    return false;
  }

  if (src->size2 != dest->size2) {
    std::cerr << "InitFromAdjacencyMatrix --> src->size2 != dest->size2" << std::endl;
    return false;
  }

  gsl_matrix_set_zero(dest);

  for (int iRow = 0; iRow < src->size1; ++iRow) {
    double sum = 0.0;

    for (int iCol = 0; iCol < src->size2; ++iCol)
      sum += gsl_matrix_get(src, iRow, iCol);

    gsl_matrix_set(dest, iRow, iRow, sum);
  }

  return gsl_matrix_sub(dest, src) == 0;
}

double GetAverage(const gsl_vector &vec) {
  double avg = 0.0;

  for (int i = 0; i < vec.size; ++i)
    avg += gsl_vector_get(&vec, i);

  return avg / vec.size;
}

void GetProcessesDistribution(const gsl_vector &vec, double avg) {
  std::string firstSubgraph = "first subgraph: ";
  std::string secondSubgraph = "second subgraph: ";

  for (int i = 0; i < vec.size; ++i) {
    if (gsl_vector_get(&vec, i) < avg)
      firstSubgraph += "Q" + std::to_string(i + 1) + " ";
    else
      secondSubgraph += "Q" + std::to_string(i + 1) + " ";
  }

  std::cout << firstSubgraph << std::endl;
  std::cout << secondSubgraph << std::endl;
}

int main() {
  GslSquareMatrixWrapper adjacencyMatrixWrapper(adjacencyMatrixDim, adjacencyMatrix);
  GslSquareMatrixWrapper laplasMatrixWrapper(adjacencyMatrixDim);
  gsl_matrix *adjMatrix = adjacencyMatrixWrapper.GetMatrix();
  gsl_matrix *laplasMatrix = laplasMatrixWrapper.GetMatrix();

  if (!InitFromAdjacencyMatrix(laplasMatrix, adjMatrix))
    return 1;

  std::cout << "Laplas matrix: " << std::endl;
  laplasMatrixWrapper.Dump(std::cout);
  std::cout << std::endl;

  EigenSquareSymMatrixSolver solver(laplasMatrix, adjacencyMatrixDim);

  if (!solver.Solve()) {
    std::cout << "main --> an error occured while calculating eigen values" << std::endl;
    return 2;
  }

  int iSecondEigen = 1;
  gsl_vector_view secondEigenVector = solver.GetEigenVector(iSecondEigen);

  std::cout << "Second eigen vector: " << std::endl;
  gsl_vector_fprintf(stdout, &secondEigenVector.vector, "%g");
  std::cout << std::endl;

  double eigenVectorAverage = GetAverage(secondEigenVector.vector);
  std::cout << "avg = " << eigenVectorAverage << std::endl;
  GetProcessesDistribution(secondEigenVector.vector, eigenVectorAverage);
  return 0;
}
