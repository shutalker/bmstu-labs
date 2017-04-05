#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <new>


using namespace std;


extern double deltaR;
extern double deltaFi;
extern double R;
extern double r;
extern double robenConditionValue;         //Числовые значения
extern double neumannConditionValue;       //правых частей
extern double dirichletConditionValue;     //граничных условий
extern double rightPartOfEquations;        //правая часть уравнений


class LAEmatrix
{
    private:
        double **LAEkoeffMatrix;
        double  *solutionVector;
        int      nodesInWidthAmount;
        int      nodesInCircleAmount;
        int      nodesAmount;
        int      stepMultiple;

        double * generateInnerNodeKoeffs(const double);
        int      fillMatrixWithInnerKoeffs(int &);
        int      fillMatrixWithRobenKoeffs(int &);
        int      fillMatrixWithNeumannKoeffs(int &);
        int      fillMatrixWithDirichletKoeffs(int &);
    
    public:
        LAEmatrix(int);
        ~LAEmatrix();
        void   fillMatrixWithKoeffs();
        void   gaussLAESolution();
        void   showLAEkoeffMatrix();
        void   makeApproximation(const LAEmatrix *);
        void   solutionOutput();
};
