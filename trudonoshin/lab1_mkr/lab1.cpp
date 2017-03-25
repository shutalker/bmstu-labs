#include "lab1.h"


int main(int argc, char **argv)
{
	LAEmatrix matrix;

    matrix.fillMatrixWithKoeffs();
    matrix.gaussLAESolution();
    matrix.showSolution();
	
	exit(0);
}
