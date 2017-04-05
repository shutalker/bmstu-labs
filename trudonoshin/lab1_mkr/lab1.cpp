#include "lab1.h"


int main(int argc, char **argv)
{
	LAEmatrix baseMatrix(2);
    LAEmatrix apprxMatrix(1);

    baseMatrix.fillMatrixWithKoeffs();
    baseMatrix.gaussLAESolution();
    apprxMatrix.fillMatrixWithKoeffs();
    apprxMatrix.gaussLAESolution();
    apprxMatrix.makeApproximation(&baseMatrix);
    apprxMatrix.solutionOutput();
	
	exit(0);
}
