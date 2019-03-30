#ifndef LAB4_MLP_APPROX_NET_H
#define LAB4_MLP_APPROX_NET_H

#include <vector>
#include <iostream>
#include <fstream>
using size_t = std::size_t;
typedef std::pair<std::vector<double>, double> TrainPair;

class MLPApproxNet {
public:
    struct Config {
        size_t hiddenLayerNeurons;
        size_t inVecDim;
        int    trainEpochs;

        struct ActivationFunction {
            double sigma;
        };

        struct RProp {
            double alpha  = 1.5;
            double beta   = 0.5;
            double nuMin  = 1e-6;
            double nuMax  = 50.0;
            double nuInit = 0.1;
        };

        ActivationFunction activationFunction;
        RProp rprop;

        void Dump();
    };

    void   Init(const Config &conf, bool randomizeWeights = true);
    void   Train(const std::vector<TrainPair> &trainSet);
    double Test(const std::vector<double> &inVec) noexcept(false);

private:
    const size_t OUT_LAYER_NEURONS = 1;
    Config config;

    struct NeuronLayer {
        size_t weightVecDim;
        std::vector<double> weights;
        std::vector<double> prevDerivatives;
        std::vector<double> derivatives;
        std::vector<double> trainCoeffs;
        std::vector<double> weightedSums;
        std::vector<double> outputs;

        void Init(size_t neurons, size_t vecDim);
        void RandomizeWeights(size_t neurons);
    };

    NeuronLayer hiddenLayer;
    NeuronLayer outLayer;

    double Signum(double x);
    double ActivationFunction(double weightedSum);
    double ActivationFunctionDerivative(double weightedSum);
    void   OfflineTrain(const std::vector<TrainPair> &trainSet);
    void   SetHiddenLayerWeightedSums(const std::vector<double> &trainVec);
    void   SetHiddenLayerOutputs();
    double GetNeuralNetOutput();

    void AdjustHiddenLayerWeightDerivatives(const std::vector<double> &trainVec,
        double trainError);

    void AdjustOutLayerWeightDerivatives(double trainError);

    double GetWeightTrainCoeff(double prevTrainCoeff, double weightDerivative,
        double prevWeightDerivative);

    void AdjustHiddenLayerWeights();
    void AdjustOutLayerWeights();
};

#endif // LAB4_MLP_APPROX_NET_H
