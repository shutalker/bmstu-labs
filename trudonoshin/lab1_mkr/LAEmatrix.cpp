#include "LAEmatrix.h"


LAEmatrix::LAEmatrix(int multiple)
{
    int i, j;
    double  dR, dFi;

    stepMultiple = multiple;
    dR = deltaR / (double)(multiple);
    dFi = deltaFi / (double)(multiple);
    nodesInWidthAmount = (int)((fabs(R - r) / dR) + 1);
	nodesInCircleAmount = (int)((2 * M_PI) / dFi);
	nodesAmount = nodesInWidthAmount * nodesInCircleAmount;

    solutionVector = NULL;
    LAEkoeffMatrix = new double * [nodesAmount];

	for(i = 0; i < nodesAmount; i++)
	{
		LAEkoeffMatrix[i] = new double [nodesAmount + 1];

		for(j = 0; j <= nodesAmount; j++)
			LAEkoeffMatrix[i][j] = 0;
	}
}


LAEmatrix::~LAEmatrix()
{
    int i;

	for(i = 0; i <= nodesAmount; i++)
		delete [] LAEkoeffMatrix[i];

	delete [] LAEkoeffMatrix;
    if(solutionVector != NULL)
        delete [] solutionVector;

    nodesInWidthAmount = nodesInCircleAmount = nodesAmount = 0;
}


double * LAEmatrix::generateInnerNodeKoeffs(const double radius)
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
	int innerNodesInWidth = nodesInWidthAmount - 2;
	int koeffPos_1, koeffPos_2, koeffPos_3, koeffPos_4, koeffPos_5;
	double radius = r;
	double *innerNodeKoeffs;

	for(j = 0; j < innerNodesInWidth; j++)
	{
		radius += deltaR;
		innerNodeKoeffs = generateInnerNodeKoeffs(radius);

		for(i; i < ((j + 1) * nodesInCircleAmount); i++, offset++)
		{
			koeffPos_1 = i;
			koeffPos_2 = i + nodesInCircleAmount;
			koeffPos_3 = koeffPos_2 + 1;

			if(koeffPos_3 >= ((j + 2) * nodesInCircleAmount))
				koeffPos_3 -= nodesInCircleAmount;

			koeffPos_4 = koeffPos_2 - 1;

			if(koeffPos_4 < ((j + 1) * nodesInCircleAmount))
				koeffPos_4 += nodesInCircleAmount;

			koeffPos_5 = koeffPos_2 + nodesInCircleAmount;

			LAEkoeffMatrix[offset][koeffPos_1] = innerNodeKoeffs[0];
			LAEkoeffMatrix[offset][koeffPos_2] = innerNodeKoeffs[1];
			LAEkoeffMatrix[offset][koeffPos_3] = innerNodeKoeffs[2];
			LAEkoeffMatrix[offset][koeffPos_4] = innerNodeKoeffs[3];
			LAEkoeffMatrix[offset][koeffPos_5] = innerNodeKoeffs[4];
            LAEkoeffMatrix[offset][nodesAmount] = rightPartOfEquations;
		}
		delete [] innerNodeKoeffs;
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

    for(i = 0; i < nodesInCircleAmount; i++, offset++)
    {
        koeffPos_1 = i;
	    koeffPos_2 = i + nodesInCircleAmount;
        LAEkoeffMatrix[offset][koeffPos_1] = robenKoeffs[0];
		LAEkoeffMatrix[offset][koeffPos_2] = robenKoeffs[1];
        LAEkoeffMatrix[offset][nodesAmount] = robenConditionValue;
    }

    return offset;
}


int LAEmatrix::fillMatrixWithNeumannKoeffs(int &offset)
{
    int i;
    int startPos, nodesOffset;
    int koeffPos_1, koeffPos_2;
    double neumannKoeffs[2];

    neumannKoeffs[0] = 1 / deltaR;  //T62, T72, ...
    neumannKoeffs[1] = -1 / deltaR; //T63, T73, ...

    startPos = (nodesInWidthAmount - 2) * nodesInCircleAmount;
    nodesOffset  = (int)(nodesInCircleAmount / 2);

    for(i = nodesOffset; i < nodesInCircleAmount; i++, offset++)
    {
        koeffPos_1 = i + startPos;
        koeffPos_2 = koeffPos_1 + nodesInCircleAmount;
        LAEkoeffMatrix[offset][koeffPos_1] = neumannKoeffs[0];
		LAEkoeffMatrix[offset][koeffPos_2] = neumannKoeffs[1];
        LAEkoeffMatrix[offset][nodesAmount] = neumannConditionValue;
    }

    return offset;
}


int LAEmatrix::fillMatrixWithDirichletKoeffs(int &offset)
{
    int i, j, startPos, endPos;
    double dirichletKoeff = 1.0; //T03, T13, ...


    startPos = (nodesInWidthAmount - 1) * nodesInCircleAmount;
    endPos = startPos + (int)(nodesInCircleAmount / 2);

    for(i = startPos; i < endPos; i++, offset++)
    {

        LAEkoeffMatrix[offset][i] = dirichletKoeff;
        LAEkoeffMatrix[offset][nodesAmount] = dirichletConditionValue;
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

    for(diagRow = 0, diagCol = 0; diagRow < nodesAmount && diagCol < nodesAmount; diagRow++, diagCol++)
    {
        currentRow = diagRow;

        for(i = diagRow; i < nodesAmount; i++)
            if(fabs(LAEkoeffMatrix[i][diagCol]) > fabs(LAEkoeffMatrix[currentRow][diagCol]))
                currentRow = i;

        if(currentRow != diagRow)
        {
            for(j = diagCol; j <= nodesAmount; j++)
            {
                double tempElement = LAEkoeffMatrix[currentRow][j];

                LAEkoeffMatrix[currentRow][j] = LAEkoeffMatrix[diagRow][j];
                LAEkoeffMatrix[diagRow][j] = tempElement;
            }
        }

        for(i = 0; i < nodesAmount; i++)
        {
            if(i != diagRow)
            {
                gaussKoeff = LAEkoeffMatrix[i][diagCol] / LAEkoeffMatrix[diagRow][diagCol];
                for(j = 0; j <= nodesAmount; j++)
                    LAEkoeffMatrix[i][j] -= LAEkoeffMatrix[diagRow][j] * gaussKoeff;
            }
        }
    }

    solutionVector = new double [nodesAmount];

    for(i = 0; i < nodesAmount; i++)
        solutionVector[i] = LAEkoeffMatrix[i][nodesAmount] / LAEkoeffMatrix[i][i];
}


void LAEmatrix::solutionOutput()
{
    int i, j = 0, k = 0;
    short shiftFlag = 0;
    double x, y;

    ofstream fout("data.txt");

    for(i = 0; i < nodesAmount ; i++)
    {

        x = (r+k*deltaR) * cos(j*deltaFi);
        y = (r+k*deltaR) * sin(j*deltaFi);

        fout << x << "    " << y << "    " << solutionVector[i] << endl;

        if(shiftFlag)
        {
          fout << endl << x << "    " << y << "    " << solutionVector[i] <<endl;
          shiftFlag = 0;
        }

        if(j && (j % (nodesInCircleAmount - 1) == 0))
        {
            j = 0;
            shiftFlag = 1;
            k++;
        }
        else
            j++;
    }

    fout.close();
}


void LAEmatrix::makeApproximation(const LAEmatrix *matrix)
{
    int matchingNodesStep;
    int i, j;
    double h1, h2, hRatioInPowerOfK;
    double approximatedSolution;

    matchingNodesStep = matrix->stepMultiple / stepMultiple;
    h1 = deltaR / matrix->stepMultiple;
    h2 = deltaR / stepMultiple;
    hRatioInPowerOfK = pow((h1/h2), 2);

    for(i = 0, j = 0; i < nodesAmount; i++, j += matchingNodesStep)
    {
        approximatedSolution = (solutionVector[j] * hRatioInPowerOfK - matrix->solutionVector[j]) / (hRatioInPowerOfK - 1);
        solutionVector[i] = approximatedSolution;
    }
}


void LAEmatrix::showLAEkoeffMatrix()
{
    int i, j;

	for(i = 0; i < nodesAmount; i++)
	{
		for(j = 0; j <= nodesAmount; j++)
			cout << LAEkoeffMatrix[i][j] << " ";
		cout << endl;
	}
}
