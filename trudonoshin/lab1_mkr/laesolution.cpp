#include "laesolution.h"


double * gaussLAESolution(const int dimention)
{
    int i, j;
    int diagRow, diagCol;
    int currentRow;
    double gaussKoeff;
    double *solutionVector;

    for(diagRow = 0, diagCol = 0; diagRow < dimention && diagCol < dimention; diagRow++, diagCol++)
    {
        currentRow = diagRow;
        
        for(i = diagRow; i < dimention; i++)
            if(fabs(koeffMatrix[i][diagCol]) > fabs(koeffMatrix[currentRow][diagCol]))
                currentRow = i;

        if(currentRow != diagRow)
        {
            for(j = diagCol; j <= dimention; j++)
            {
                double tempElement = koeffMatrix[currentRow][j];
                
                koeffMatrix[currentRow][j] = koeffMatrix[diagRow][j];
                koeffMatrix[diagRow][j] = tempElement;
            }
        }

        for(i = 0; i < dimention; i++)
        {
            if(i != diagRow)
            {
                gaussKoeff = koeffMatrix[i][diagCol] / koeffMatrix[diagRow][diagCol];
                for(j = 0; j <= dimention; j++)
                    koeffMatrix[i][j] -= koeffMatrix[diagRow][j] * gaussKoeff;
            }
        }
    }

    solutionVector = new double [dimention];

    for(i = 0; i < dimention; i++)
        solutionVector[i] = koeffMatrix[i][dimention] / koeffMatrix[i][i];

    return solutionVector;
}
