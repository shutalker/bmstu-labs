#include <iostream>
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
        int      unitsInWidthAmount;
        int      unitsInCircleAmount;
        int      unitsAmount;

        double * generateInnerUnitKoeffs(const double );
        int      fillMatrixWithInnerKoeffs(int &);
        int      fillMatrixWithRobenKoeffs(int &);
        int      fillMatrixWithNeumannKoeffs(int &);
        int      fillMatrixWithDirichletKoeffs(int &);
    
    public:
        LAEmatrix();
        ~LAEmatrix();
        void   fillMatrixWithKoeffs();
        void   gaussLAESolution();
        void   showLAEkoeffMatrix();
        void   showSolution();
};
