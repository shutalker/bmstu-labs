#include "LAEmatrix.h"


LAEmatrix::LAEmatrix()
{
    int i, j;

    unitsInWidthAmount = (int)((fabs(R - r) / deltaR) + 1);
	unitsInCircleAmount = (int)((2 * M_PI) / deltaFi);
	unitsAmount = unitsInWidthAmount * unitsInCircleAmount;

    LAEkoeffMatrix = new double * [unitsAmount];

	for(i = 0; i < unitsAmount; i++)
	{
		LAEkoeffMatrix[i] = new double [unitsAmount + 1];

		for(j = 0; j <= unitsAmount; j++)
			LAEkoeffMatrix[i][j] = 0;
	}
}


LAEmatrix::~LAEmatrix()
{
    int i;  

	for(i = 0; i <= unitsAmount; i++)
		delete [] LAEkoeffMatrix[i];

	delete [] LAEkoeffMatrix;
    delete [] solutionVector;

    unitsInWidthAmount = unitsInCircleAmount = unitsAmount = 0;
}


double * LAEmatrix::generateInnerUnitKoeffs(const double radius)
{
    double *koeffs = new double [5];
	double deltaR_2 = pow(deltaR, 2);
	double deltaFi_2 = pow(deltaFi, 2);
	double radius_2 = pow(radius, 2);

	koeffs[0] = (2 * radius - deltaR) / (2 * radius * deltaR_2); //T_00
	koeffs[1] = -2 * (radius_2 * deltaFi_2 + deltaR_2) / (radius_2 * deltaR_2 * deltaFi_2); //T_01
	koeffs[2] = 1 / (radius_2 * deltaFi_2); //T_11
	koeffs[3] = koeffs[2];	                //T_11_1
	koeffs[4] = (2 * radius + deltaR) / (2 * radius * deltaR_2); //T_02

	return koeffs;  
}


int LAEmatrix::fillMatrixWithInnerKoeffs(int &offset)
{
	int j, i = 0;
	int innerUnitsInWidth = unitsInWidthAmount - 2;
	int koeffPos_1, koeffPos_2, koeffPos_3, koeffPos_4, koeffPos_5;
	double radius = r;
	double *innerUnitKoeffs;

	for(j = 0; j < innerUnitsInWidth; j++)
	{
		radius += deltaR;
		innerUnitKoeffs = generateInnerUnitKoeffs(radius);

		for(i; i < ((j + 1) * unitsInCircleAmount); i++, offset++)
		{
			koeffPos_1 = i;
			koeffPos_2 = i + unitsInCircleAmount;
			koeffPos_3 = koeffPos_2 + 1;

			if(koeffPos_3 >= ((j + 2) * unitsInCircleAmount))
				koeffPos_3 -= unitsInCircleAmount;

			koeffPos_4 = koeffPos_2 - 1;

			if(koeffPos_4 < ((j + 1) * unitsInCircleAmount))
				koeffPos_4 += unitsInCircleAmount;

			koeffPos_5 = koeffPos_2 + unitsInCircleAmount;

			LAEkoeffMatrix[offset][koeffPos_1] = innerUnitKoeffs[0];
			LAEkoeffMatrix[offset][koeffPos_2] = innerUnitKoeffs[1];
			LAEkoeffMatrix[offset][koeffPos_3] = innerUnitKoeffs[2];
			LAEkoeffMatrix[offset][koeffPos_4] = innerUnitKoeffs[3];
			LAEkoeffMatrix[offset][koeffPos_5] = innerUnitKoeffs[4];
            LAEkoeffMatrix[offset][unitsAmount] = rightPartOfEquations;
		}
		delete [] innerUnitKoeffs;	
	}

    return offset;
}


int LAEmatrix::fillMatrixWithRobenKoeffs(int &offset)
{
    int i;
    int koeffPos_1, koeffPos_2;
    double robenKoeffs[2];

    robenKoeffs[0] = -(1 + deltaR) / deltaR; //T00, T10, ...
    robenKoeffs[1] = 1 / deltaR;             //T01, T11, ...

    for(i = 0; i < unitsInCircleAmount; i++, offset++)
    {
        koeffPos_1 = i;
	    koeffPos_2 = i + unitsInCircleAmount;
        LAEkoeffMatrix[offset][koeffPos_1] = robenKoeffs[0];
		LAEkoeffMatrix[offset][koeffPos_2] = robenKoeffs[1];
        LAEkoeffMatrix[offset][unitsAmount] = robenConditionValue;
    }

    return offset;
}


int LAEmatrix::fillMatrixWithNeumannKoeffs(int &offset)
{
    int i;
    int startPos, unitsOffset;
    int koeffPos_1, koeffPos_2;
    double neumannKoeffs[2];

    neumannKoeffs[0] = 1 / deltaR;  //T62, T72, ...
    neumannKoeffs[1] = -1 / deltaR; //T63, T73, ...

    startPos = (unitsInWidthAmount - 2) * unitsInCircleAmount;
    unitsOffset  = (int)(unitsInCircleAmount / 2);

    for(i = unitsOffset; i < unitsInCircleAmount; i++, offset++)
    {
        koeffPos_1 = i + startPos;
        koeffPos_2 = koeffPos_1 + unitsInCircleAmount;
        LAEkoeffMatrix[offset][koeffPos_1] = neumannKoeffs[0];
		LAEkoeffMatrix[offset][koeffPos_2] = neumannKoeffs[1];
        LAEkoeffMatrix[offset][unitsAmount] = neumannConditionValue;
    }

    return offset;
}


int LAEmatrix::fillMatrixWithDirichletKoeffs(int &offset)
{
    int i, j, startPos, endPos;
    double dirichletKoeff = 1.0; //T03, T13, ...


    startPos = (unitsInWidthAmount - 1) * unitsInCircleAmount;
    endPos = startPos + (int)(unitsInCircleAmount / 2);

    for(i = startPos; i < endPos; i++, offset++)
    {

        LAEkoeffMatrix[offset][i] = dirichletKoeff;
        LAEkoeffMatrix[offset][unitsAmount] = dirichletConditionValue;
    }

    return offset;
}


void LAEmatrix::fillMatrixWithKoeffs()
{
    int rowOffset = 0;
    
    fillMatrixWithInnerKoeffs(rowOffset);         //Уравнение теплопроводности для внутр. узлов
	fillMatrixWithRobenKoeffs(rowOffset);         //ГУ 3 рода на внутр. границе трубки
    fillMatrixWithNeumannKoeffs(rowOffset);       //ГУ 2 рода на внеш. границе нижней половины трубки
    fillMatrixWithDirichletKoeffs(rowOffset);     //ГУ 1 рода на внеш. границе верхней половины трубки
}


void LAEmatrix::gaussLAESolution()
{
    int i, j;
    int diagRow, diagCol;
    int currentRow;
    double gaussKoeff;

    for(diagRow = 0, diagCol = 0; diagRow < unitsAmount && diagCol < unitsAmount; diagRow++, diagCol++)
    {
        currentRow = diagRow;
        
        for(i = diagRow; i < unitsAmount; i++)
            if(fabs(LAEkoeffMatrix[i][diagCol]) > fabs(LAEkoeffMatrix[currentRow][diagCol]))
                currentRow = i;

        if(currentRow != diagRow)
        {
            for(j = diagCol; j <= unitsAmount; j++)
            {
                double tempElement = LAEkoeffMatrix[currentRow][j];
                
                LAEkoeffMatrix[currentRow][j] = LAEkoeffMatrix[diagRow][j];
                LAEkoeffMatrix[diagRow][j] = tempElement;
            }
        }

        for(i = 0; i < unitsAmount; i++)
        {
            if(i != diagRow)
            {
                gaussKoeff = LAEkoeffMatrix[i][diagCol] / LAEkoeffMatrix[diagRow][diagCol];
                for(j = 0; j <= unitsAmount; j++)
                    LAEkoeffMatrix[i][j] -= LAEkoeffMatrix[diagRow][j] * gaussKoeff;
            }
        }
    }

    solutionVector = new double [unitsAmount];

    for(i = 0; i < unitsAmount; i++)
        solutionVector[i] = LAEkoeffMatrix[i][unitsAmount] / LAEkoeffMatrix[i][i];
}


void LAEmatrix::showSolution()
{
    int i, j = 0, k = 0;

    for(i = 0; i < unitsAmount; i++)
    {
        cout << "T[" << j << "][" << k << "] = " << solutionVector[i] << endl;
        if(j && (j % (unitsInCircleAmount - 1) == 0))
        {
            j = 0;
            k++;
        }
        else
            j++;
    }
}


void LAEmatrix::showLAEkoeffMatrix()
{
    int i, j;

	for(i = 0; i < unitsAmount; i++)
	{
		for(j = 0; j <= unitsAmount; j++)
			cout << LAEkoeffMatrix[i][j] << " ";
		cout << endl;
	}
}
