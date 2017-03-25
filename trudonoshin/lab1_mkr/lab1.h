#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "memalloc.h"
#include "matrixgen.h"
#include "laesolution.h"


using namespace std;


double deltaR = 0.25;
double deltaFi = M_PI / 24;
double R = 50.0;
double r = 44.0;
double robenConditionValue = 0.0;         //Числовые значения
double neumannConditionValue = 0.0;       //правых частей
double dirichletConditionValue = 10.0;    //граничных условий
double rightPartOfEquations = 0.0;        //правая часть уравнений
double **koeffMatrix;
