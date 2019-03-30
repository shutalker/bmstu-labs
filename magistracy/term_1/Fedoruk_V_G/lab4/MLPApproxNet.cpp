#include "MLPApproxNet.h"
#include <iomanip>
#include <random>
#include <algorithm>

void MLPApproxNet::NeuronLayer::Init(size_t neurons, size_t vecDim) {
    weightVecDim = vecDim + 1;
    size_t dim = neurons * weightVecDim;
    weights.resize(dim);
    prevDerivatives.resize(dim);
    derivatives.resize(dim);
    trainCoeffs.resize(dim);
    weightedSums.resize(neurons);
    outputs.resize(neurons);
}

void MLPApproxNet::NeuronLayer::RandomizeWeights(size_t neurons) {
    double weightInf = -2.0 / sqrt(neurons);
    double weightSup = -weightInf;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(weightInf, weightSup);

    for (size_t iWeight = 0; iWeight < weights.size(); ++iWeight)
        weights[iWeight] = dis(gen);
}

void MLPApproxNet::Config::Dump() {
    std::cout << "config = {" << std::endl;
    std::cout << "  hiddenLayerNeurons = " << hiddenLayerNeurons << std::endl;
    std::cout << "  inVecDim = " << inVecDim << std::endl;
    std::cout << "  trainEpochs = " << trainEpochs << std::endl;
    std::cout << "  activationFunction = {" << std::endl;
    std::cout << "    sigma = " << activationFunction.sigma << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << "  rprop = {" << std::endl;
    std::cout << "    alpha = " << rprop.alpha << std::endl;
    std::cout << "    beta = " << rprop.beta << std::endl;
    std::cout << "    nuInit = " << rprop.nuInit << std::endl;
    std::cout << "    nuMin = " << rprop.nuMin << std::endl;
    std::cout << "    nuMax = " << rprop.nuMax << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << "}" << std::endl;
}

void MLPApproxNet::Init(const Config &conf, bool randomizeWeights) {
    config = conf;
    std::cout << "hiddenLayer.weightVecDim = " << hiddenLayer.weightVecDim << std::endl;
    std::cout << "outLayer.weightVecDim = " << outLayer.weightVecDim << std::endl;
    hiddenLayer.Init(config.hiddenLayerNeurons, config.inVecDim);
    outLayer.Init(OUT_LAYER_NEURONS, config.hiddenLayerNeurons);

    if (randomizeWeights) {
        hiddenLayer.RandomizeWeights(config.hiddenLayerNeurons);
        outLayer.RandomizeWeights(OUT_LAYER_NEURONS);
    }
}

double MLPApproxNet::Signum(double x) {
    if (x > 0.0)
        return 1.0;

    if (x < 0.0)
        return -1.0;

    return 0.0;
}

double MLPApproxNet::ActivationFunction(double weightedSum) {
    return 1.0 / (1.0 + exp(-config.activationFunction.sigma * weightedSum));
}

double MLPApproxNet::ActivationFunctionDerivative(double weightedSum) {
    double activationFunc = ActivationFunction(weightedSum);
    return config.activationFunction.sigma * activationFunc * (1.0 - activationFunc);
}

// TODO optimise: do not create weightedSums vector each time
void MLPApproxNet::SetHiddenLayerWeightedSums(
        const std::vector<double> &trainVec) {
    for (size_t iNeuron = 0; iNeuron < config.hiddenLayerNeurons; ++iNeuron) {
        size_t iWeight = 0;

        for (; iWeight < trainVec.size(); ++iWeight) {
            hiddenLayer.weightedSums[iNeuron] += hiddenLayer.weights[iWeight + iNeuron * hiddenLayer.weightVecDim]
                * trainVec[iWeight];
        }

        // sum neuron polarization input
        hiddenLayer.weightedSums[iNeuron] += hiddenLayer.weights[iWeight + iNeuron * hiddenLayer.weightVecDim];
    }
}

void MLPApproxNet::SetHiddenLayerOutputs() {
    for (size_t iNeuron = 0; iNeuron < config.hiddenLayerNeurons; ++iNeuron)
        hiddenLayer.outputs[iNeuron] = ActivationFunction(hiddenLayer.weightedSums[iNeuron]);
}

double MLPApproxNet::GetNeuralNetOutput() {
    double netOutput = 0;
    size_t iNeuron = 0;

    for (; iNeuron < config.hiddenLayerNeurons; ++iNeuron)
        netOutput += hiddenLayer.outputs[iNeuron] * outLayer.weights[iNeuron];

    netOutput += outLayer.weights[iNeuron];

    return netOutput;
}

void MLPApproxNet::AdjustHiddenLayerWeightDerivatives(const std::vector<double> &trainVec,
        double trainError) {
    for (size_t iNeuron = 0; iNeuron < config.hiddenLayerNeurons; ++iNeuron) {
        size_t iWeight = 0;

        for (; iWeight < trainVec.size(); ++iWeight) {
            hiddenLayer.derivatives[iWeight + iNeuron * hiddenLayer.weightVecDim] += trainError
                * outLayer.weights[iNeuron] * trainVec[iWeight]
                * ActivationFunctionDerivative(hiddenLayer.weightedSums[iNeuron]);
        }

        // adjust neuron polarization weight derivative
        hiddenLayer.derivatives[iWeight + iNeuron * hiddenLayer.weightVecDim] += trainError
            * outLayer.weights[iNeuron]
            * ActivationFunctionDerivative(hiddenLayer.weightedSums[iNeuron]);
    }
}

void MLPApproxNet::AdjustOutLayerWeightDerivatives(double trainError) {
    size_t iNeuron = 0;

    for (; iNeuron < config.hiddenLayerNeurons; ++iNeuron)
        outLayer.derivatives[iNeuron] += trainError * hiddenLayer.outputs[iNeuron];

    outLayer.derivatives[iNeuron] += trainError;
}

double MLPApproxNet::GetWeightTrainCoeff(double prevTrainCoeff, double weightDerivative,
        double prevWeightDerivative) {
    double trainCoeff = prevTrainCoeff;
    double derivativeSign = weightDerivative * prevWeightDerivative;

    if (derivativeSign > 0)
        trainCoeff = std::min(config.rprop.alpha * prevTrainCoeff, config.rprop.nuMax);
    
    if (derivativeSign < 0)
        trainCoeff = std::max(config.rprop.beta * prevTrainCoeff, config.rprop.nuMin);

    return trainCoeff;
}

void MLPApproxNet::AdjustHiddenLayerWeights() {
    size_t weightInd;
    double trainCoeff;

    for (size_t iNeuron = 0; iNeuron < config.hiddenLayerNeurons; ++iNeuron) {
        for (size_t iWeight = 0; iWeight < hiddenLayer.weightVecDim; ++iWeight) {
            weightInd = iWeight + iNeuron * hiddenLayer.weightVecDim;
            trainCoeff = GetWeightTrainCoeff(
                hiddenLayer.trainCoeffs[weightInd],
                hiddenLayer.derivatives[weightInd],
                hiddenLayer.prevDerivatives[weightInd]
            );
            hiddenLayer.weights[weightInd] -= trainCoeff * Signum(hiddenLayer.derivatives[weightInd]);
            hiddenLayer.trainCoeffs[weightInd] = trainCoeff;
        }
    }
}

void MLPApproxNet::AdjustOutLayerWeights() {
    double trainCoeff;

    for (size_t iWeight = 0; iWeight < outLayer.weightVecDim; ++iWeight) {
            trainCoeff = GetWeightTrainCoeff(
                outLayer.trainCoeffs[iWeight],
                outLayer.derivatives[iWeight],
                outLayer.prevDerivatives[iWeight]
            );
            outLayer.weights[iWeight] -= trainCoeff * Signum(hiddenLayer.derivatives[iWeight]);
            outLayer.trainCoeffs[iWeight] = trainCoeff;
    }
}

void MLPApproxNet::OfflineTrain(const std::vector<TrainPair> &trainSet) {
    double netOutput;
    double trainError;

    std::fill(hiddenLayer.derivatives.begin(), hiddenLayer.derivatives.end(), 0.0);
    std::fill(outLayer.derivatives.begin(), outLayer.derivatives.end(), 0.0);

    for (size_t iTrainPair = 0; iTrainPair < trainSet.size(); ++iTrainPair) {
        const std::vector<double> &trainVec = trainSet[iTrainPair].first;
        double trainOutput = trainSet[iTrainPair].second;

        netOutput = Test(trainVec);
        trainError = netOutput - trainOutput;

        AdjustHiddenLayerWeightDerivatives(trainVec, trainError);
        AdjustOutLayerWeightDerivatives(trainError);
    }

    AdjustHiddenLayerWeights();
    AdjustOutLayerWeights();

    hiddenLayer.prevDerivatives = hiddenLayer.derivatives;
    outLayer.prevDerivatives = outLayer.derivatives;
}

void MLPApproxNet::Train(const std::vector<TrainPair> &trainSet) {
    config.Dump();
    std::fill(hiddenLayer.trainCoeffs.begin(), hiddenLayer.trainCoeffs.end(),
        config.rprop.nuInit);
    std::fill(outLayer.trainCoeffs.begin(), outLayer.trainCoeffs.end(),
        config.rprop.nuInit);

    std::fill(hiddenLayer.prevDerivatives.begin(), hiddenLayer.prevDerivatives.end(), 0.0);
    std::fill(outLayer.prevDerivatives.begin(), outLayer.prevDerivatives.end(), 0.0);

    for (size_t iEpoch = 0; iEpoch < config.trainEpochs; ++iEpoch) {
        std::cout << "Train epoch: " << iEpoch << std::endl;
        OfflineTrain(trainSet);
    }
}

double MLPApproxNet::Test(const std::vector<double> &inVec) {
    if (inVec.size() != config.inVecDim) {
        throw std::string("MLPApproxNet::OfflineTrain --> invalid "\
            "train vector size");
    }

    std::fill(hiddenLayer.weightedSums.begin(), hiddenLayer.weightedSums.end(), 0.0);
    SetHiddenLayerWeightedSums(inVec);
    SetHiddenLayerOutputs();
    return GetNeuralNetOutput();
}
