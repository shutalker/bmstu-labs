#include "memalloc.h" 


double ** allocateKoeffMatrix(const int dimention)
{
	int i, j;
	double **tempMatrix;

	tempMatrix = new double* [dimention];
	for(i = 0; i < dimention; i++)
	{
		tempMatrix[i] = new double [dimention + 1];

		for(j = 0; j < dimention + 1; j++)
			tempMatrix[i][j] = 0;
	}

	return tempMatrix;
}


void deallocateKoeffMatrix(const int dimention)
{
	int i;

	for(i = 0; i < dimention + 1; i++)
		delete [] koeffMatrix[i];

	delete [] koeffMatrix;
}
