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


int main(int argc, char **argv)
{
	int unitsInWidthAmount;
	int unitsInCircleAmount;
	int unitsAmount;

	unitsInWidthAmount = (int)(((R - r) / deltaR) + 1);
	unitsInCircleAmount = (int)((2 * M_PI) / deltaFi);
	unitsAmount = unitsInWidthAmount * unitsInCircleAmount;
	koeffMatrix = allocateKoeffMatrix(unitsAmount);
    fillMatrixWithKoeffs(unitsInWidthAmount, unitsInCircleAmount, unitsAmount);
	printMatrix(koeffMatrix, unitsAmount);
	deallocateKoeffMatrix(unitsAmount);
	exit(0);
}
