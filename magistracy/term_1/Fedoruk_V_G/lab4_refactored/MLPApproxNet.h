#ifndef _LAB4_REFACTORED_MLP_APPROX_NET_H_
#define _LAB4_REFACTORED_MLP_APPROX_NET_H_

#include <vector>

typedef std::pair<std::vector<double>, double> TrainPair;
using size_t = std::size_t;

class MLPApproxNet {
public:
    struct Config {
        size_t inputLayerNeurons;
        size_t inputVecDim;
        size_t epochs;
        double sigma;
        double trainCoeff;
    };

    void   Init(const Config &conf);
    void   OnlineTrain(const std::vector<TrainPair> &trainSet);
    double Test(const std::vector<double> &inVec);
    void   DumpSigmoids(const std::vector<TrainPair> &vecSet);
    
private:
    Config config;
    size_t inputWeightDim;
    size_t outputWeightDim;
    std::vector<double> inputLayerWeights;
    std::vector<double> inputLayerOutputs;
    std::vector<double> outputLayerWeights;

    void   RandomizeWeights();
    double ActivationFunction(double weightedSum);
    double ActivationFunctionDerivative(double weightedSum);
    void   SetInputLayerOutputs(const std::vector<double> &inVec);
    double GetNeuralNetOutput();
    void   AdjustWeightsByBPROP(double trainError, const std::vector<double> &inVec);
};

#endif // _LAB4_REFACTORED_MLP_APPROX_NET_H_
