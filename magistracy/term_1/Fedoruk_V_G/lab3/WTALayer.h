#ifndef LAB3_WTA_LAYER_H
#define LAB3_WTA_LAYER_H

#include <vector>
#include <iostream>
#include <fstream>
using size_t = std::size_t;

class WTALayer {
public:
    struct Config {
        size_t neurons;
        size_t inVecDim;
        int    trainEpochs;
        double trainCoeff;
        double trainPenalty;
    };

    void Init(const Config &conf, bool randomizeWeights = true);
    bool SetWeights(const std::vector<double> &initialWeights);
    size_t Test(const std::vector<double> &inVec);

    void Train(const std::vector<std::vector<double>> &trainSet,
        std::ostream &output);

    void DumpWeights(std::ostream &output, int precision,
        const std::string &title = "", bool gnuplot = false);

private:
    Config config;
    std::vector<double>   weights;
    std::vector<uint64_t> winHistory;

    void RandomizeWeights();
    std::vector<double> GetWeightedSums(const std::vector<double> &inVec);
    size_t DetectWinner(const std::vector<double> &weightedSums);
    void AdjustWeights(size_t iWinner, const std::vector<double> &inVec);

    void AdjustWinHistory(size_t iWinner) { winHistory[iWinner] += 1; }
};

#endif // LAB3_WTA_LAYER_H
