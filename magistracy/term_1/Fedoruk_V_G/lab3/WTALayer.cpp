#include "WTALayer.h"
#include <iomanip>
#include <random>
#include <algorithm>

void WTALayer::Init(const Config &conf, bool randomizeWeights) {
    config = conf;

    if (!randomizeWeights)
        return;

    weights.resize(config.neurons * config.inVecDim);
    winHistory.resize(config.neurons * config.inVecDim, 0);
    RandomizeWeights();
}

bool WTALayer::SetWeights(const std::vector<double> &initialWeights) {
    if (initialWeights.size() != config.neurons * config.inVecDim) {
        std::cerr << "invalid weights size: " << initialWeights.size() << std::endl;
        return false;
    }

    weights = initialWeights;
    winHistory.resize(config.neurons * config.inVecDim, 0);

    return true;
}

void WTALayer::RandomizeWeights() {
    const double WEIGHT_INF = -1.0;
    const double WEIGHT_SUP = 1.0;

    std::random_device randomizer;
    std::mt19937 randGen(randomizer());
    std::uniform_real_distribution<> dist(WEIGHT_INF, WEIGHT_SUP);

    double weightNorm;
    double weight = 0.0;

    // randomize and normalize all weights
    for (size_t iNeuron = 0; iNeuron < config.neurons; ++iNeuron) {
        weightNorm = 0.0;

        for (size_t iWeight = 0; iWeight < config.inVecDim; ++iWeight) {
            weight = dist(randGen);
            weights[iWeight + iNeuron * config.inVecDim] = weight;
            weightNorm += weight * weight;
        }

        weightNorm = sqrt(weightNorm);

        for (size_t iWeight = 0; iWeight < config.inVecDim; ++iWeight)
            weights[iWeight + iNeuron * config.inVecDim] /= weightNorm;
    }
}

std::vector<double> WTALayer::GetWeightedSums(const std::vector<double> &inVec) {
    std::vector<double> weightedSums(config.neurons, 0.0);
    
    for (size_t iNeuron = 0; iNeuron < config.neurons; ++iNeuron) {
        for (size_t iWeight = 0; iWeight < config.inVecDim; ++iWeight) {
            weightedSums[iNeuron] += weights[iWeight + iNeuron * config.inVecDim] * inVec[iWeight];
            // TODO code refactoring
            weightedSums[iNeuron] -= config.trainPenalty * winHistory[iNeuron];
        }
    }

    return weightedSums;
}

size_t WTALayer::DetectWinner(const std::vector<double> &weightedSums) {
    return std::distance(weightedSums.begin(),
        std::max_element(weightedSums.begin(), weightedSums.end()));
}

size_t WTALayer::Test(const std::vector<double> &inVec) {
    if (inVec.size() != config.inVecDim)
        throw std::string("WTALayer::Test --> invalid input vector size!");

    return DetectWinner(GetWeightedSums(inVec));
}

void WTALayer::AdjustWeights(size_t iWinner, const std::vector<double> &inVec) {
    double prevWeight;
    double currWeight;
    double weightNorm = 0.0;

    for (size_t iWeight = 0; iWeight < config.inVecDim; ++iWeight) {
        prevWeight = weights[iWeight + iWinner * config.inVecDim];
        currWeight = prevWeight + config.trainCoeff * (inVec[iWeight] - prevWeight);
        weights[iWeight + iWinner * config.inVecDim] = currWeight;
        weightNorm += currWeight * currWeight;
    }

    weightNorm = sqrt(weightNorm);

    // normalize weights
    for (size_t iWeight = 0; iWeight < config.inVecDim; ++iWeight) {
        weights[iWeight + iWinner * config.inVecDim] /= weightNorm;
    }
}

void WTALayer::Train(const std::vector<std::vector<double>> &trainSet,
        std::ostream &output) {
    size_t iWinner;

    for (size_t iEpoch = 0; iEpoch < config.trainEpochs; ++iEpoch) {
        std::cout << "Train epoch: " << iEpoch << std::endl;

        for (size_t iVec = 0; iVec < trainSet.size(); ++iVec) {
            iWinner = Test(trainSet[iVec]);
            AdjustWinHistory(iWinner);
            AdjustWeights(iWinner, trainSet[iVec]);

            DumpWeights(output, 16, "", true);
        }
    }
}

void WTALayer::DumpWeights(std::ostream &output, int precision,
        const std::string &title, bool gnuplot) {
    if (!gnuplot)
        output << title << std::endl;

    if (gnuplot) {
        for (size_t iWeight = 0; iWeight < weights.size(); ++iWeight)
            output << 0.0 << "\t";

        output << std::endl;
    }

    for (size_t iWeight = 0; iWeight < weights.size(); ++iWeight)
        output << std::setprecision(precision) << weights[iWeight] << "\t";

    output << std::endl;

    if (gnuplot)
        output << std::endl << std::endl;
}
