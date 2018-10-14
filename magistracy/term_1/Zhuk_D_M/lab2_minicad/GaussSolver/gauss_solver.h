#ifndef LAB2_MINICAD_GAUSS_SOLVER_H
#define LAB2_MINICAD_GAUSS_SOLVER_H

#include <vector>
#include <map>
#include <memory>

typedef std::vector<double> Vector;
typedef std::shared_ptr<Vector> VectorPtr;


// test struct
struct ExtendedMatrix {
    VectorPtr src;
    int rows = 0;
    int cols = 0;

    void Clear() {
        src.reset();
        rows = 0;
        cols = 0;
    }
};

class BasicGaussSolver {
public:
    BasicGaussSolver() {}
    virtual ~BasicGaussSolver() {}
    virtual void Solve() = 0;

private:
    BasicGaussSolver(const BasicGaussSolver &other) = delete;
};

class GaussSolver: public BasicGaussSolver {
public:
    GaussSolver() {}
    GaussSolver(const ExtendedMatrix &matrix);

    void Init(const ExtendedMatrix &matrix);
    void Clear();
    void DumpMatrix();
    VectorPtr GetSolution() { return solution; }

    virtual void Solve();

private:
    ExtendedMatrix matrix;
    VectorPtr solution;
    std::map<int, int> transmutations;

    GaussSolver(const GaussSolver &other) = delete;

    void ForwardTraverse();
    void BackwardTraverse();

    bool IsZero(double val);
    bool IsEqual(double x, double y);
    bool SwapRow(int iRow);
    double ExpressFreeMember(int iRow);
};

#endif // #ifndef LAB2_MINICAD_GAUSS_SOLVER_H