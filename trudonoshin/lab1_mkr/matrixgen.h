#include <cmath>
#include <new>


extern double deltaR;
extern double deltaFi;
extern double r;
extern double **koeffMatrix;
extern double robenConditionValue;         //Числовые значения
extern double neumannConditionValue;       //правых частей
extern double dirichletConditionValue;     //граничных условий
extern double rightPartOfEquations;        //правая часть уравнений

double * generateInnerUnitKoeffs(const double );
int fillMatrixWithInnerKoeffs(const int , const int );
int fillMatrixWithRobenKoeffs(const int , const int, const int );
int fillMatrixWithNeumannKoeffs(const int , const int, const int );
int fillMatrixWithDirichletKoeffs(const int , const int, const int );
void fillMatrixWithKoeffs(const int , const int , const int );
