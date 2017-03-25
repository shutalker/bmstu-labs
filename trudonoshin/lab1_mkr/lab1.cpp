#include "lab1.h"


void printMatrix(double **matrix, const int dimention)
{
	int i, j;

	for(i = 0; i < dimention; i++)
	{
		for(j = 0; j < dimention + 1; j++)
			cout << matrix[i][j] << " ";
		cout << endl;
	}

}


void printSolutionVector(double * const vector, const int dimention, const int unitsInCircle)
{
    int i, j = 0, k = 0;

    for(i = 0; i < dimention; i++)
    {
        cout << "T[" << j << "][" << k << "] = " << vector[i] << endl;
        if(j && (j % (unitsInCircle - 1) == 0))
        {
            j = 0;
            k++;
        }
        else
            j++;
    }
}


int main(int argc, char **argv)
{
	int unitsInWidthAmount;
	int unitsInCircleAmount;
	int unitsAmount;
    double *solution;

	unitsInWidthAmount = (int)(((R - r) / deltaR) + 1);
	unitsInCircleAmount = (int)((2 * M_PI) / deltaFi);
	unitsAmount = unitsInWidthAmount * unitsInCircleAmount;
	koeffMatrix = allocateKoeffMatrix(unitsAmount);
    fillMatrixWithKoeffs(unitsInWidthAmount, unitsInCircleAmount, unitsAmount);
    solution = gaussLAESolution(unitsAmount);
    printSolutionVector(solution, unitsAmount, unitsInCircleAmount);
//	printMatrix(koeffMatrix, unitsAmount);
	deallocateKoeffMatrix(unitsAmount);
	exit(0);
}
