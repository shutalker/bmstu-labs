#include "MLPApproxNet.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

void MLPApproxNet::Init(const Config &conf) {
    config = conf;
    inputWeightDim = config.inputVecDim + 1;
    outputWeightDim = config.inputLayerNeurons + 1;
    inputLayerWeights.resize(inputWeightDim * config.inputLayerNeurons);
    inputLayerOutputs.resize(config.inputLayerNeurons);
    outputLayerWeights.resize(outputWeightDim);

    RandomizeWeights();
}

void MLPApproxNet::RandomizeWeights() {
    const double WEIGHT_SUP = 0.5;
    const double WEIGHT_INF = -0.5;

    std::random_device randomizer;
    std::mt19937 randGen(randomizer());
    std::uniform_real_distribution<> dist(WEIGHT_INF, WEIGHT_SUP);

    size_t iNeuron = 0;
    size_t inputWeightInd;

    for (; iNeuron < config.inputLayerNeurons; ++iNeuron) {
        // input layer weights initialization
        for (size_t iWeight = 0; iWeight < inputWeightDim; ++iWeight) {
            inputWeightInd = iWeight + iNeuron * inputWeightDim;
            inputLayerWeights[inputWeightInd] = dist(randGen);
        }

        // output layer weights initialization
        outputLayerWeights[iNeuron] = dist(randGen);
    }

    //output layer polarizator initializing
    outputLayerWeights[iNeuron] = 0.0;
}

double MLPApproxNet::ActivationFunction(double weightedSum) {
    return 1.0 / (1.0 + exp(-config.sigma * weightedSum));
}

double MLPApproxNet::ActivationFunctionDerivative(double activationFunction) {
    return config.sigma * activationFunction * (1.0 - activationFunction);
}

void MLPApproxNet::SetInputLayerOutputs(const std::vector<double> &inVec) {
    size_t iWeight;
    size_t weightInd;

    // calculating weighted sums for input layer
    for (size_t iNeuron = 0; iNeuron < config.inputLayerNeurons; ++iNeuron) {
        for (iWeight = 0; iWeight < config.inputVecDim; ++iWeight) {
            weightInd = iWeight + iNeuron * inputWeightDim;
            inputLayerOutputs[iNeuron] += inputLayerWeights[weightInd] * inVec[iWeight];
        }

        // calculate weighted sum for input layer polarizator
        weightInd = iWeight + iNeuron * inputWeightDim;
        inputLayerOutputs[iNeuron] += inputLayerWeights[weightInd];
    }

    // calculating outputs for input layers
    for (size_t iNeuron = 0; iNeuron < config.inputLayerNeurons; ++iNeuron)
        inputLayerOutputs[iNeuron] = ActivationFunction(inputLayerOutputs[iNeuron]);
}

double MLPApproxNet::GetNeuralNetOutput() {
    size_t iNeuron = 0;
    double output = 0.0;

    for (; iNeuron < config.inputLayerNeurons; ++iNeuron)
        output += inputLayerOutputs[iNeuron] * outputLayerWeights[iNeuron];
    
    // sum output layer polarizator contribution
    output += outputLayerWeights[iNeuron];
    return output;
}

double MLPApproxNet::Test(const std::vector<double> &inVec) {
    std::fill(inputLayerOutputs.begin(), inputLayerOutputs.end(), 0.0);
    SetInputLayerOutputs(inVec);
    return GetNeuralNetOutput();
}

void MLPApproxNet::AdjustWeightsByBPROP(double trainError,
        const std::vector<double> &inVec) {
    size_t iNeuron;
    size_t iWeight;
    size_t weightInd;
    double weightDerivative;
    double backPropError;

    for (iNeuron = 0; iNeuron < config.inputLayerNeurons; ++iNeuron) {
        backPropError = trainError * outputLayerWeights[iNeuron];
        backPropError *= ActivationFunctionDerivative(inputLayerOutputs[iNeuron]);

        for (iWeight = 0; iWeight < config.inputVecDim; ++iWeight) {
            weightInd = iWeight + iNeuron * inputWeightDim;
            weightDerivative = backPropError * inVec[iWeight];
            inputLayerWeights[weightInd] -= config.trainCoeff * weightDerivative;
        }

        // adjusting input layer polarizator weight
        weightInd = iWeight + iNeuron * inputWeightDim;
        inputLayerWeights[weightInd] -= config.trainCoeff * backPropError;

        // adjusting output layer weights
        weightDerivative = trainError * inputLayerOutputs[iNeuron];
        outputLayerWeights[iNeuron] -= config.trainCoeff * weightDerivative;
    }

    // adjusting output layer polarizator weight
    outputLayerWeights[iNeuron] -= config.trainCoeff * trainError;
}

void MLPApproxNet::DumpSigmoids(const std::vector<TrainPair> &vecSet) {
    double s;
    std::vector<std::vector<std::pair<double, double>>> sigmoids(config.inputLayerNeurons);

    for (size_t iVec = 0; iVec < vecSet.size(); ++iVec) {
        const std::vector<double> &trainVec = vecSet[iVec].first;
        Test(trainVec);
            
        for (size_t iNeuron = 0; iNeuron < config.inputLayerNeurons; ++iNeuron) {
            s = inputLayerOutputs[iNeuron] * outputLayerWeights[iNeuron];
            sigmoids[iNeuron].emplace_back(std::make_pair(trainVec[0], s));
        }
    }

    std::string filename;

    for (size_t iNeuron = 0; iNeuron < config.inputLayerNeurons; ++iNeuron) {
        filename = std::to_string(iNeuron) + ".txt";
        std::ofstream out(filename, std::ios::binary);

        for (size_t iPoint = 0; iPoint < sigmoids[iNeuron].size(); ++iPoint) {
             out << std::setprecision(15) << sigmoids[iNeuron][iPoint].first
                << "\t" << sigmoids[iNeuron][iPoint].second << std::endl;
        }
    }
}

void MLPApproxNet::OnlineTrain(const std::vector<TrainPair> &trainSet) {
    double netOutput;
    double trainOutput;
    double trainError;

    for (size_t iEpoch = 0; iEpoch < config.epochs; ++iEpoch) {
        for (size_t iVec = 0; iVec < trainSet.size(); ++iVec) {
            const std::vector<double> &trainVec = trainSet[iVec].first;
            trainOutput = trainSet[iVec].second;
            netOutput = Test(trainVec);
            trainError = netOutput - trainOutput;
            AdjustWeightsByBPROP(trainError, trainVec);
        }
    }
}
