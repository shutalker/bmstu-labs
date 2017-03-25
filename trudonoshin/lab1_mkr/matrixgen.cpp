#include "matrixgen.h"


double * generateInnerUnitKoeffs(const double radius)
{
	double *koeffs = new double [5];
	double deltaR_2 = pow(deltaR, 2);
	double deltaFi_2 = pow(deltaFi, 2);
	double radius_2 = pow(radius, 2);

	koeffs[0] = (2 * radius - deltaR) / (2 * radius * deltaR_2); //T_00
	koeffs[1] = -2 * (radius_2 * deltaFi_2 + deltaR_2) / (radius_2 * deltaR_2 * deltaFi_2); //T_01
	koeffs[2] = 1 / (radius_2 * deltaFi_2); //T_11
	koeffs[3] = koeffs[2];	//T_11_1
	koeffs[4] = (2 * radius + deltaR) / (2 * radius * deltaR_2); //T_02

	return koeffs;
}


int fillMatrixWithInnerKoeffs(const int unitsInWidth, const int unitsInCircle)
{
	int j, i = 0;
	int rowOffset = 0;
	int innerUnitsInWidth = unitsInWidth - 2;
    int rightPartKoeffPos = unitsInWidth * unitsInCircle;
	int koeffPos_1, koeffPos_2, koeffPos_3, koeffPos_4, koeffPos_5;
	double radius = r;
	double *innerUnitKoeffs;

	for(j = 0; j < innerUnitsInWidth; j++)
	{
		radius += deltaR;
		innerUnitKoeffs = generateInnerUnitKoeffs(radius);

		for(i; i < ((j + 1) * unitsInCircle); i++, rowOffset++)
		{
			koeffPos_1 = i;
			koeffPos_2 = i + unitsInCircle;
			koeffPos_3 = koeffPos_2 + 1;

			if(koeffPos_3 >= ((j + 2) * unitsInCircle))
				koeffPos_3 -= unitsInCircle;

			koeffPos_4 = koeffPos_2 - 1;

			if(koeffPos_4 < ((j + 1) * unitsInCircle))
				koeffPos_4 += unitsInCircle;

			koeffPos_5 = koeffPos_2 + unitsInCircle;

			koeffMatrix[rowOffset][koeffPos_1] = innerUnitKoeffs[0];
			koeffMatrix[rowOffset][koeffPos_2] = innerUnitKoeffs[1];
			koeffMatrix[rowOffset][koeffPos_3] = innerUnitKoeffs[2];
			koeffMatrix[rowOffset][koeffPos_4] = innerUnitKoeffs[3];
			koeffMatrix[rowOffset][koeffPos_5] = innerUnitKoeffs[4];
            koeffMatrix[rowOffset][rightPartKoeffPos] = rightPartOfEquations;
		}
		delete [] innerUnitKoeffs;	
	}

    return rowOffset;
}


int fillMatrixWithRobenKoeffs(const int offset, const int unitsInCircle, const int unitsAmount)
{
    int i;
    int rowOffset = offset;
    int koeffPos_1, koeffPos_2;
    double robenKoeffs[2];

    robenKoeffs[0] = -(1 + deltaR) / deltaR; //T00, T10, ...
    robenKoeffs[1] = 1 / deltaR;             //T01, T11, ...

    for(i = 0; i < unitsInCircle; i++, rowOffset++)
    {
        koeffPos_1 = i;
	    koeffPos_2 = i + unitsInCircle;
        koeffMatrix[rowOffset][koeffPos_1] = robenKoeffs[0];
		koeffMatrix[rowOffset][koeffPos_2] = robenKoeffs[1];
        koeffMatrix[rowOffset][unitsAmount] = robenConditionValue;
    }

    return rowOffset;
}


int fillMatrixWithNeumannKoeffs(const int offset, const int unitsInCircle, const int unitsAmount)
{
    int i, startPos;
    int rowOffset = offset;
    int unitsOffset = (int)(unitsInCircle / 2);
    int koeffPos_1, koeffPos_2;
    double neumannKoeffs[2];

    neumannKoeffs[0] = 1 / deltaR;  //T62, T72, ...
    neumannKoeffs[1] = -1 / deltaR; //T63, T73, ...

    startPos = 2 * unitsInCircle;

    for(i = unitsOffset; i < unitsInCircle; i++, rowOffset++)
    {
        koeffPos_1 = i + startPos;
        koeffPos_2 = koeffPos_1 + unitsInCircle;
        koeffMatrix[rowOffset][koeffPos_1] = neumannKoeffs[0];
		koeffMatrix[rowOffset][koeffPos_2] = neumannKoeffs[1];
        koeffMatrix[rowOffset][unitsAmount] = neumannConditionValue;
    }

    return rowOffset;
}


int fillMatrixWithDirichletKoeffs(const int offset, const int unitsInCircle, const int unitsAmount)
{
    int i, startPos, endPos;
    int rowOffset = offset;
    int koeffPos_1;
    double dirichletKoeff = 1.0; //T03, T13, ...


    startPos = 2 * unitsInCircle;
    endPos = (int)(unitsInCircle / 2);

    for(i = 0; i < endPos; i++, rowOffset++)
    {
        koeffPos_1 = i + startPos;
        koeffMatrix[rowOffset][koeffPos_1] = dirichletKoeff;
        koeffMatrix[rowOffset][unitsAmount] = dirichletConditionValue;
    }

    return rowOffset;
}


void fillMatrixWithKoeffs(const int unitsInWidthAmount, const int unitsInCircleAmount, const int unitsAmount)
{
    int rowOffset = 0;
    
    rowOffset = fillMatrixWithInnerKoeffs(unitsInWidthAmount, unitsInCircleAmount);         //Уравнение теплопроводности для внутр. узлов
	rowOffset = fillMatrixWithRobenKoeffs(rowOffset, unitsInCircleAmount, unitsAmount);     //ГУ 3 рода на внутр. границе трубки
    rowOffset = fillMatrixWithNeumannKoeffs(rowOffset, unitsInCircleAmount, unitsAmount);   //ГУ 2 рода на внеш. границе нижней половины трубки
    rowOffset = fillMatrixWithDirichletKoeffs(rowOffset, unitsInCircleAmount, unitsAmount); //ГУ 1 рода на внеш. границе верхней половины трубки
}
